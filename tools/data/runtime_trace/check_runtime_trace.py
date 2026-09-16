"""Executable CPU contracts for coarse memory regions and component lineage."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .extract import (
    RUN,
    Bindings,
    build_graph,
    conflicting_writes,
    entry_read_regions,
    gemm_accesses,
    launch_configuration,
    normalize_leases,
    subtract_regions,
    union,
    version_dependencies,
    view_regions,
)
from .lineage import trace_best
from .match import compare
from .service_replay import _inline_schema_graph


def node(identity, seq, reads=(), writes=(), stream=0, **extra):
    return {
        "id": identity,
        "seq": seq,
        "stream": stream,
        "kind": "kernel",
        "implementation": "body",
        "configuration": {"block": [32, 1, 1]},
        "evidence": {"name": identity},
        "unknowns": [],
        "footprint_complete": True,
        "reads": [
            {
                "buffer": b,
                "regions": [[lo, hi]],
                "entry_regions": [[lo, hi]],
                "port": None,
                "evidence": "observed_global_memory_runs",
            }
            for b, lo, hi in reads
        ],
        "writes": [
            {
                "buffer": b,
                "regions": [[lo, hi]],
                "conflicting_regions": [],
                "atomic_unknown": False,
                "port": None,
                "evidence": "observed_global_memory_runs",
            }
            for b, lo, hi in writes
        ],
        **extra,
    }


def buffers():
    return [
        {"id": name, "base": i * 128 + 1024, "bytes": 64, "roles": roles, "first_seq": 0, "last_seq": None}
        for i, (name, roles) in enumerate([("a", ["input:0"]), ("b", []), ("c", ["output:0"])])
    ]


def graph(nodes):
    allocations = buffers()
    edges, versions = version_dependencies(nodes, allocations)
    return {
        "schema": "coarse-component-graph/v1",
        "context": {"task": "task", "input_signature": "input", "environment_signature": "env"},
        "nodes": nodes,
        "buffers": allocations,
        "edges": edges,
        "versions": versions,
    }


class Regions(unittest.TestCase):
    def test_exact_region_subtraction_preserves_holes(self):
        self.assertEqual(subtract_regions([[0, 16], [24, 32]], [[4, 8], [12, 28]]), [[0, 4], [8, 12], [28, 32]])

    def test_merged_region_split_at_storage_boundaries_and_gaps(self):
        binding = Bindings(
            [
                {"id": "a", "base": 100, "bytes": 4, "first_seq": 0, "last_seq": None},
                {"id": "b", "base": 104, "bytes": 4, "first_seq": 0, "last_seq": None},
                {"id": "c", "base": 112, "bytes": 4, "first_seq": 0, "last_seq": None},
            ],
            1,
        )
        self.assertEqual(
            [(a["id"] if a else None, o, n) for a, o, n in binding.pieces(102, 14)],
            [("a", 2, 2), ("b", 0, 4), (None, 0, 4), ("c", 0, 4)],
        )

    def test_union_keeps_holes(self):
        self.assertEqual(union([(0, 4), (4, 8), (12, 16)]), [[0, 8], [12, 16]])

    def test_read_then_own_write_uses_entry(self):
        entry, unknown = entry_read_regions([(0, 128, 0, 4, 0)], [(0, 128, 0, 4, 1)])
        self.assertEqual(entry, [[0, 128]])
        self.assertFalse(unknown)

    def test_own_write_then_read_is_internal(self):
        entry, unknown = entry_read_regions([(0, 128, 0, 4, 2)], [(0, 128, 0, 4, 1)])
        self.assertFalse(entry)
        self.assertEqual(unknown, [[0, 128]])

    def test_other_thread_order_never_from_log(self):
        entry, unknown = entry_read_regions([(0, 4, 0, 4, 0)], [(0, 4, 1, 4, 100)])
        self.assertFalse(entry)
        self.assertEqual(unknown, [[0, 4]])

    def test_partial_internal_write_preserves_other_entry_bytes(self):
        entry, unknown = entry_read_regions([(0, 8, 0, 8, 2)], [(4, 8, 0, 4, 1)])
        self.assertEqual(entry, [[0, 4]])
        self.assertEqual(unknown, [[4, 8]])

    def test_racing_writers(self):
        self.assertEqual(conflicting_writes([(0, 4, 0, 4, 0), (0, 4, 1, 4, 1)]), [[0, 4]])

    def test_same_thread_overwrite_not_race(self):
        self.assertFalse(conflicting_writes([(0, 4, 0, 4, 0), (0, 4, 0, 4, 1)]))

    def test_view_stride(self):
        self.assertEqual(
            view_regions({"shape": [3], "stride": [2], "offset_bytes": 4, "element_size": 4}),
            [[4, 8], [12, 16], [20, 24]],
        )

    def test_transposed_dense_view(self):
        self.assertEqual(
            view_regions({"shape": [4, 3], "stride": [1, 4], "offset_bytes": 0, "element_size": 4}), [[0, 48]]
        )

    def test_live_wrappers_alias(self):
        a = {"id": "a", "base": 100, "bytes": 16, "first_seq": 0, "last_seq": 5, "roles": ["input:0"]}
        b = {"id": "b", "base": 104, "bytes": 8, "first_seq": 1, "last_seq": 6, "roles": ["output:0"]}
        self.assertEqual(len(normalize_leases([a, b])), 1)

    def test_address_reuse_is_new_generation(self):
        a = {"id": "a", "base": 100, "bytes": 16, "first_seq": 0, "last_seq": 5, "roles": []}
        b = {"id": "b", "base": 100, "bytes": 16, "first_seq": 5, "last_seq": None, "roles": []}
        self.assertEqual(len(normalize_leases([a, b])), 2)


class Versions(unittest.TestCase):
    def test_inline_unknown_input_is_preserved_on_consumer(self):
        g = graph([node("r", 1, reads=[("b", 0, 64)])])
        original = copy.deepcopy(g)
        inline = _inline_schema_graph(g)
        self.assertEqual(g, original)
        self.assertFalse(inline["edges"])
        self.assertIn("input_provenance_unresolved", inline["nodes"][0]["unknowns"])
        self.assertEqual(inline["nodes"][0]["evidence"]["unknown_input_dependencies"], original["edges"])

    def test_inline_proven_null_source_is_rejected(self):
        g = graph([node("r", 1, reads=[("b", 0, 64)])])
        g["edges"][0]["certainty"] = "proven_region_dependency"
        with self.assertRaisesRegex(ValueError, "invalid_unknown_dependency"):
            _inline_schema_graph(g)

    def test_unknown_stream_mutation_needs_observed_sync(self):
        ns = [node("w", 2, writes=[("b", 0, 64)]), node("r", 3, reads=[("b", 0, 64)])]
        self.assertEqual(
            version_dependencies(ns, buffers(), [{"seq": 0}])[0][0]["certainty"], "partial_or_ambiguous_dependency"
        )
        self.assertEqual(
            version_dependencies(ns, buffers(), [{"seq": 0}], [{"seq": 1, "action": "device_sync"}])[0][0][
                "certainty"
            ],
            "proven_region_dependency",
        )

    def test_incomplete_consumer_does_not_prove_initial_version(self):
        ns = [node("r", 1, reads=[("a", 0, 64)], footprint_complete=False)]
        self.assertEqual(version_dependencies(ns, buffers())[0][0]["certainty"], "possible_initial_value_read")

    def test_unknown_intervening_call_can_overwrite(self):
        ns = [
            node("w", 0, writes=[("b", 0, 64)]),
            node("unknown", 1, footprint_complete=False),
            node("r", 2, reads=[("b", 0, 64)]),
        ]
        self.assertEqual(version_dependencies(ns, buffers())[0][0]["certainty"], "partial_or_ambiguous_dependency")

    def test_unknown_intervening_call_invalidates_initial_value(self):
        ns = [
            node("unknown", 1, footprint_complete=False, unknowns=["unsupported_memory_instruction"]),
            node("r", 2, reads=[("a", 0, 64)]),
        ]
        edge = version_dependencies(ns, buffers())[0][0]
        self.assertEqual(edge["certainty"], "entry_value_unobserved")
        self.assertIsNone(edge["source"])
        self.assertIsNone(edge["version"])

    def test_complete_overwrite_recovers_after_unknown(self):
        ns = [
            node("unknown", 0, footprint_complete=False),
            node("w", 1, writes=[("b", 0, 64)]),
            node("r", 2, reads=[("b", 0, 64)]),
        ]
        self.assertEqual(version_dependencies(ns, buffers())[0][0]["certainty"], "proven_region_dependency")

    def test_partial_overwrite_recovers_only_written_region(self):
        ns = [
            node("old", 0, writes=[("b", 0, 64)]),
            node("unknown", 1, footprint_complete=False),
            node("w", 2, writes=[("b", 16, 32)]),
            node("r", 3, reads=[("b", 0, 64)]),
        ]
        edges, _ = version_dependencies(ns, buffers())
        self.assertEqual(
            {e["source"]: e["certainty"] for e in edges},
            {"old": "partial_or_ambiguous_dependency", "w": "proven_region_dependency"},
        )

    def test_partial_overwrite_has_two_producers(self):
        ns = [
            node("w0", 0, writes=[("b", 0, 64)]),
            node("w1", 1, writes=[("b", 16, 32)]),
            node("r", 2, reads=[("b", 0, 64)]),
        ]
        edges, _ = version_dependencies(ns, buffers())
        self.assertEqual({e["source"]: e["regions"] for e in edges}, {"w0": [[0, 16], [32, 64]], "w1": [[16, 32]]})

    def test_unordered_streams_remain_unknown(self):
        ns = [node("w", 0, writes=[("b", 0, 64)], stream=1), node("r", 3, reads=[("b", 0, 64)], stream=2)]
        edges, _ = version_dependencies(ns, buffers())
        self.assertEqual(edges[0]["certainty"], "unordered_stream_writers")

    def test_event_orders_streams(self):
        ns = [node("w", 0, writes=[("b", 0, 64)], stream=1), node("r", 3, reads=[("b", 0, 64)], stream=2)]
        orders = [
            {"seq": 1, "action": "record", "event": 8, "stream": 1},
            {"seq": 2, "action": "wait", "event": 8, "stream": 2},
        ]
        edges, _ = version_dependencies(ns, buffers(), orders=orders)
        self.assertEqual(edges[0]["certainty"], "proven_region_dependency")

    def test_destroyed_event_not_reused_as_order(self):
        ns = [node("w", 0, writes=[("b", 0, 64)], stream=1), node("r", 4, reads=[("b", 0, 64)], stream=2)]
        orders = [
            {"seq": 1, "action": "record", "event": 8, "stream": 1},
            {"seq": 2, "action": "destroy", "event": 8},
            {"seq": 3, "action": "wait", "event": 8, "stream": 2},
        ]
        self.assertEqual(
            version_dependencies(ns, buffers(), orders=orders)[0][0]["certainty"], "unordered_stream_writers"
        )

    def test_device_sync_orders_following_read(self):
        ns = [node("w", 0, writes=[("b", 0, 64)], stream=1), node("r", 2, reads=[("b", 0, 64)], stream=2)]
        self.assertEqual(
            version_dependencies(ns, buffers(), orders=[{"seq": 1, "action": "device_sync"}])[0][0]["source"], "w"
        )

    def test_truncation_never_proves_dependency(self):
        ns = [node("w", 0, writes=[("b", 0, 64)], footprint_complete=False), node("r", 2, reads=[("b", 0, 64)])]
        self.assertEqual(version_dependencies(ns, buffers())[0][0]["certainty"], "partial_or_ambiguous_dependency")

    def test_unknown_memory_operation_degrades_edge(self):
        ns = [node("w", 0, writes=[("b", 0, 64)]), node("r", 2, reads=[("b", 0, 64)])]
        self.assertEqual(
            version_dependencies(ns, buffers(), [{"seq": 1}])[0][0]["certainty"], "partial_or_ambiguous_dependency"
        )


class LibraryContracts(unittest.TestCase):
    def parameters(self, **changes):
        return {
            "state_query_ok": True,
            "alpha": 1,
            "beta": 0,
            "ta": 0,
            "tb": 1,
            "m": 2,
            "n": 4,
            "k": 3,
            "a": 100,
            "b": 200,
            "c": 300,
            "lda": 2,
            "ldb": 4,
            "ldc": 2,
            "at": 0,
            "bt": 0,
            "ct": 0,
            **changes,
        }

    def test_column_major_regions(self):
        self.assertEqual(
            gemm_accesses(self.parameters()),
            [("a", "read", [(100, 124)]), ("b", "read", [(200, 248)]), ("c", "write", [(300, 332)])],
        )

    def test_padding_not_falsely_read(self):
        self.assertEqual(gemm_accesses(self.parameters(lda=4))[0][2], [(100, 108), (116, 124), (132, 140)])

    def test_zero_alpha_does_not_read_ab(self):
        self.assertEqual(gemm_accesses(self.parameters(alpha=0)), [("c", "write", [(300, 332)])])

    def test_beta_reads_prior_c(self):
        self.assertIn(("c", "read", [(300, 332)]), gemm_accesses(self.parameters(beta=1)))

    def test_unknown_device_scalar_not_assumed(self):
        with self.assertRaises(ValueError):
            gemm_accesses(self.parameters(alpha=None))


class Correspondence(unittest.TestCase):
    def test_digest_collision_is_rejected_even_with_asserts_disabled(self):
        other = graph([node("different", 1, reads=[("a", 4, 64)], writes=[("c", 0, 64)])])
        with patch("tools.data.runtime_trace.match.digest", return_value="collision"):
            with self.assertRaisesRegex(ValueError, "digest collision"):
                compare(self.g, other)

    def test_fused_role_revision_keeps_identity_separate_from_version(self):
        other = copy.deepcopy(self.g)
        other["nodes"][0]["implementation"] = "revised-fused-body"
        other["nodes"][0]["configuration"]["block"] = [256, 1, 1]
        match = compare(self.g, other)["matches"][0]
        self.assertTrue(match["identity_evidence_complete"])
        self.assertEqual(match["version_relation"], "revised")
        result = trace_best(
            [
                {"turn": 1, "correct": False, "score": 0, "graph": self.g},
                {"turn": 2, "correct": True, "score": 1, "graph": other},
            ]
        )
        self.assertEqual(result["components"][0]["earliest_observed_call_role_candidate"]["turn"], 1)
        self.assertIsNone(result["components"][0]["earliest_observed_retained_match"])

    def test_fusion_does_not_match_a_separate_stage(self):
        split = graph(
            [
                node("a", 1, reads=[("a", 0, 64)], writes=[("b", 0, 64)]),
                node("b", 2, reads=[("b", 0, 64)], writes=[("c", 0, 64)]),
            ]
        )
        self.assertFalse(compare(split, self.g)["matches"])

    def test_layout_change_does_not_nominate_same_identity(self):
        other = copy.deepcopy(self.g)
        other["buffers"][0]["views"] = [{"shape": [4, 4], "stride": [1, 4], "dtype": "float32"}]
        self.assertFalse(compare(self.g, other)["matches"])

    def test_unobserved_input_does_not_certify_role_identity(self):
        other = copy.deepcopy(self.g)
        other["edges"][0].update(source=None, certainty="entry_value_unobserved")
        result = compare(other, other)
        self.assertFalse(result["matches"][0]["identity_evidence_complete"])
        self.assertNotEqual(result["matches"][0]["relation"], "retained_observed_component")

    def test_extended_launch_configuration_missing_is_unknown(self):
        self.assertTrue(launch_configuration({})[1])
        self.assertTrue(launch_configuration({"launch_num_attrs": 1})[1])
        self.assertFalse(launch_configuration({"launch_num_attrs": 0})[1])

    def test_unknown_config_preserves_structure_not_retention(self):
        for unknown in ["opaque_launch_configuration", "opaque_workspace_configuration"]:
            other = copy.deepcopy(self.g)
            other["nodes"][0]["unknowns"].append(unknown)
            self.assertEqual(
                compare(self.g, other)["matches"][0]["relation"], "same_regional_structure_configuration_unknown"
            )

    def test_no_data_effect_call_is_separate(self):
        empty = graph([node("empty", 1)])
        self.assertEqual(
            compare(empty, empty)["matches"][0]["relation"], "no_observed_data_effect_call_correspondence"
        )
        result = trace_best([{"turn": 1, "correct": True, "score": 1, "graph": empty}])
        self.assertEqual(result["components"], [])
        self.assertEqual(result["no_observed_data_effect_calls"], ["empty"])

    def setUp(self):
        self.g = graph([node("first", 1, reads=[("a", 0, 64)], writes=[("c", 0, 64)])])

    def test_names_and_trace_ids_do_not_align_calls(self):
        other = graph([node("renamed", 12, reads=[("a", 0, 64)], writes=[("c", 0, 64)])])
        self.assertEqual(compare(self.g, other)["matches"][0]["relation"], "retained_observed_component")

    def test_implementation_change_after_interface_alignment(self):
        other = copy.deepcopy(self.g)
        other["nodes"][0]["implementation"] = "tiled-body"
        self.assertEqual(compare(self.g, other)["matches"][0]["relation"], "implementation_changed")

    def test_runtime_configuration_change(self):
        other = copy.deepcopy(self.g)
        other["nodes"][0]["configuration"]["alpha"] = 2
        self.assertEqual(compare(self.g, other)["matches"][0]["relation"], "configuration_changed")

    def test_repeated_calls_keep_ambiguity(self):
        other = graph(
            [
                node("x", 1, reads=[("a", 0, 64)], writes=[("c", 0, 64)]),
                node("y", 2, reads=[("a", 0, 64)], writes=[("c", 0, 64)]),
            ]
        )
        self.assertFalse(compare(self.g, other)["matches"])
        self.assertTrue(compare(self.g, other)["ambiguous"])

    def test_best_missing_not_replaced(self):
        r = trace_best(
            [{"turn": 1, "correct": True, "score": 1, "graph": self.g}, {"turn": 2, "correct": True, "score": 2}]
        )
        self.assertEqual(r["status"], "best_graph_missing")

    def test_copy_returns_original_observed_match(self):
        r = trace_best(
            [
                {"turn": 1, "correct": True, "score": 1, "graph": self.g},
                {"turn": 2, "correct": True, "score": 2, "graph": self.g},
            ]
        )
        self.assertEqual(r["components"][0]["earliest_observed_retained_match"]["turn"], 1)

    def test_failed_trajectory_has_no_membership(self):
        self.assertEqual(trace_best([{"turn": 1, "correct": False}])["components"], [])


class BoundedCapture(unittest.TestCase):
    def raw_graph(self, *, memory=False, finish=True, dropped=0, exact=True, inspected=True, regions=None):
        records = [
            {"type": "config", "schema": "coarse-memory-runs/v1", "cta_limit": -1, "capacity": 1},
            {
                "type": "launch",
                "id": 0,
                "seq": 1,
                "parent": -1,
                "stream": 0,
                "name": "opaque",
                "implementation": "0",
                "implementation_version": "version42",
                "implementation_inspected": inspected,
                "grid": [1, 1, 1],
                "block": [32, 1, 1],
                "shared": 0,
                "arguments": [],
                "launch_num_attrs": 0,
                "memory_traced": memory,
                "memory_skip_reason": "vendor_memory_opaque",
                "unsupported_memory_instructions": 0,
            },
        ]
        if regions is not None:
            records.append({"type": "record_format", "format": "exact_byte_regions/v1"})
        if finish:
            records += [
                {
                    "type": "complete",
                    "seq": 2,
                    "id": 0,
                    "runs": len(regions or []),
                    "dropped": dropped,
                    "drop_count_exact": exact,
                },
                {"type": "end", "seq": 3, "launches": 1},
            ]
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "trace"
            prefix.with_suffix(".jsonl").write_text(
                "\n".join(json.dumps(r) for r in records) + ("\n" if finish else '\n{"type":"complete","seq":')
            )
            if memory:
                Path(str(prefix) + ".launch0.bin").write_bytes(
                    b"".join(RUN.pack(address, 0, length, 1, mode, 0) for address, length, mode in regions or [])
                )
            return build_graph(prefix, buffers())

    def test_region_summary_preserves_holes(self):
        n = self.raw_graph(memory=True, regions=[(1024, 4, 1), (1032, 4, 1), (1280, 8, 2)])["nodes"][0]
        self.assertEqual(n["reads"][0]["regions"], [[0, 4], [8, 12]])
        self.assertTrue(n["footprint_complete"])

    def test_region_inplace_keeps_version_unknown(self):
        n = self.raw_graph(memory=True, regions=[(1024, 16, 1), (1028, 8, 2)])["nodes"][0]
        self.assertEqual(n["reads"][0]["entry_regions"], [[0, 4], [12, 16]])
        self.assertEqual(n["reads"][0]["internal_or_unordered_regions"], [[4, 12]])
        self.assertIn("inplace_order_requires_detailed_capture", n["unknowns"])

    def test_region_overlap_and_atomic_flags_keep_write_unknown(self):
        for flag in [4, 8]:
            n = self.raw_graph(memory=True, regions=[(1280, 4, 2), (1280, 4, flag)])["nodes"][0]
            self.assertIn("write_version_ambiguous", n["unknowns"])

    def test_region_hash_overflow_is_incomplete(self):
        n = self.raw_graph(memory=True, regions=[(1024, 4, 1)], dropped=1, exact=False)["nodes"][0]
        self.assertFalse(n["footprint_complete"])

    def test_opaque_vendor_preserves_call_version_not_fabricated_access(self):
        graph = self.raw_graph()
        n = graph["nodes"][0]
        self.assertEqual(n["implementation"], "version42")
        self.assertFalse(n["footprint_complete"])
        self.assertEqual(n["reads"], [])
        self.assertIn("vendor_memory_opaque", n["unknowns"])

    def test_saturated_counter_lower_bound_marks_incomplete(self):
        n = self.raw_graph(memory=True, dropped=1, exact=False)["nodes"][0]
        self.assertFalse(n["footprint_complete"])
        self.assertFalse(n["evidence"]["dropped_runs_exact"])

    def test_uninspected_vendor_identity_stays_unknown(self):
        n = self.raw_graph(inspected=False)["nodes"][0]
        self.assertEqual(n["implementation"], "uninspected")
        self.assertIn("opaque_implementation_configuration", n["unknowns"])

    def test_failed_process_keeps_call_and_explicit_truncated_metadata(self):
        graph = self.raw_graph(finish=False)
        self.assertEqual(len(graph["nodes"]), 1)
        self.assertFalse(graph["nodes"][0]["footprint_complete"])
        self.assertFalse(graph["coverage"]["trace_process_complete"])
        self.assertEqual(graph["coverage"]["memory_unknown_events"][0]["name"], "truncated_metadata_record")


if __name__ == "__main__":
    unittest.main()
