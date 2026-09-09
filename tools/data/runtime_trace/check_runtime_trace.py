"""Focused CPU contracts for the offline NVBit graph and lineage workflow."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from .extract import EVENT, build_graph
from .lineage import trace_best


def instruction(i, opcode, space="GLOBAL", read=False, write=False):
    return {
        "type": "instruction",
        "id": i,
        "function": "test",
        "offset": 16 * i,
        "sass": opcode,
        "opcode": opcode,
        "space": space,
        "load": read,
        "store": write,
        "size": 4,
        "mrefs": int(read or write),
        "predicate": -1,
        "predicate_uniform": False,
        "operands": [],
        "captured_constant_bytes": 0,
    }


def capture(root, launch_events, instructions, *, block=1, grid=1, ctas=-1, dropped=0, extra=()):
    p = root / "trace"
    records = [{"type": "config", "cta_limit": ctas, "capacity": 100, "max_launches": 10}] + instructions
    for lid, events in enumerate(launch_events):
        records.extend(r for r in extra if r.get("before_launch") == lid)
        records += [
            {
                "type": "launch",
                "id": lid,
                "name": "k",
                "stream": 0,
                "grid": [grid, 1, 1],
                "block": [block, 1, 1],
                "dynamic_shared": 0,
                "selected": True,
            },
            {"type": "complete", "launch": lid, "events": len(events), "dropped": dropped},
        ]
        Path(f"{p}.launch{lid}.bin").write_bytes(
            b"".join(EVENT.pack(addr, 0, i, cta, tid, pred, 1, 0) for i, addr, cta, tid, pred in events)
        )
    records.append({"type": "end"})
    Path(str(p) + ".jsonl").write_text("\n".join(json.dumps(r) for r in records))
    return build_graph(
        p, {}, [{"role": "input:0", "base": 100, "bytes": 4}, {"role": "output:0", "base": 200, "bytes": 4}]
    )


class MemoryContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_same_stream_producer(self):
        g = capture(
            self.root,
            [[(0, 200, 0, 0, 1)], [(1, 200, 0, 0, 1)]],
            [instruction(0, "STG", write=True), instruction(1, "LDG", read=True)],
        )
        self.assertTrue(
            any(e["kind"] == "memory" and e["source"] == "l0:i0" and e["target"] == "l1:i1" for e in g["edges"])
        )

    def test_racing_writer_is_not_log_order(self):
        g = capture(
            self.root,
            [[(0, 200, 0, 0, 1), (0, 200, 0, 1, 1), (1, 200, 0, 0, 1)]],
            [instruction(0, "STG", write=True), instruction(1, "LDG", read=True)],
            block=2,
        )
        self.assertFalse(any(e["kind"] == "memory" for e in g["edges"]))
        self.assertIn("memory_producer_unknown_or_racing", g["coverage"]["unknowns"])

    def test_unordered_output_not_assigned(self):
        g = capture(self.root, [[(0, 200, 0, 0, 1), (0, 200, 0, 1, 1)]], [instruction(0, "STG", write=True)], block=2)
        self.assertFalse(any(e["kind"] == "output_memory" for e in g["edges"]))

    def test_same_thread_overwrite_selects_latest(self):
        g = capture(
            self.root,
            [[(0, 200, 0, 0, 1), (1, 200, 0, 0, 1), (2, 200, 0, 0, 1)]],
            [instruction(0, "STG", write=True), instruction(1, "STG", write=True), instruction(2, "LDG", read=True)],
        )
        mem = [e for e in g["edges"] if e["kind"] == "memory"]
        self.assertEqual([(e["source"], e["target"]) for e in mem], [("l0:i1", "l0:i2")])

    def test_predicate_false_does_not_write(self):
        g = capture(
            self.root,
            [[(0, 200, 0, 0, 1), (1, 200, 0, 0, 0), (2, 200, 0, 0, 1)]],
            [instruction(0, "STG", write=True), instruction(1, "STG", write=True), instruction(2, "LDG", read=True)],
        )
        self.assertEqual([e["source"] for e in g["edges"] if e["kind"] == "memory"], ["l0:i0"])

    def test_captured_prefix_not_complete_output(self):
        g = capture(self.root, [[(0, 200, 0, 0, 1)]], [instruction(0, "STG", write=True)], dropped=2)
        self.assertFalse(any(e["kind"] == "output_memory" for e in g["edges"]))
        self.assertFalse(g["coverage"]["capture_complete"])

    def test_unseen_ctas_cannot_prove_unique_memory(self):
        g = capture(
            self.root,
            [[(0, 200, 0, 0, 1), (1, 200, 0, 0, 1)]],
            [instruction(0, "STG", write=True), instruction(1, "LDG", read=True)],
            grid=2,
            ctas=1,
        )
        self.assertFalse(any(e["kind"] == "memory" for e in g["edges"]))

    def test_host_memory_mutation_invalidates_prior(self):
        g = capture(
            self.root,
            [[(0, 200, 0, 0, 1)], [(1, 200, 0, 0, 1)]],
            [instruction(0, "STG", write=True), instruction(1, "LDG", read=True)],
            extra=[{"type": "memory_api", "before_launch": 1, "name": "cuMemcpyHtoD"}],
        )
        self.assertFalse(any(e["kind"] == "memory" for e in g["edges"]))

    def test_truncated_binary_rejected(self):
        capture(self.root, [[(0, 200, 0, 0, 1)]], [instruction(0, "STG", write=True)])
        (self.root / "trace.launch0.bin").write_bytes(b"x")
        with self.assertRaises(ValueError):
            build_graph(self.root / "trace", {})

    def test_byte_overlap_retains_both_writers(self):
        a = instruction(0, "STG", write=True)
        b = instruction(1, "STG", write=True)
        b["size"] = 2
        g = capture(
            self.root,
            [[(0, 200, 0, 0, 1), (1, 202, 0, 0, 1), (2, 200, 0, 0, 1)]],
            [a, b, instruction(2, "LDG", read=True)],
        )
        self.assertEqual(
            {(e["source"], e["attrs"]["bytes_per_observed_read"]) for e in g["edges"] if e["kind"] == "memory"},
            {("l0:i0", 2), ("l0:i1", 2)},
        )

    def test_shared_barrier_orders_cross_thread(self):
        barrier = instruction(1, "BAR.SYNC.DEFER_BLOCKING", space="NONE")
        barrier["operands"] = [{"type": "IMM_UINT64", "text": "0x0", "bytes": 8}]
        g = capture(
            self.root,
            [[(0, 0, 0, 0, 1), (0, 4, 0, 1, 1), (1, 0, 0, 0, 1), (1, 0, 0, 1, 1), (2, 4, 0, 0, 1), (2, 0, 0, 1, 1)]],
            [
                instruction(0, "STS", space="SHARED", write=True),
                barrier,
                instruction(2, "LDS", space="SHARED", read=True),
            ],
            block=2,
        )
        self.assertEqual(
            [(e["source"], e["target"]) for e in g["edges"] if e["kind"] == "memory"], [("l0:i0", "l0:i2")]
        )

    def test_missing_barrier_participant_blocks_edge(self):
        barrier = instruction(1, "BAR.SYNC", space="NONE")
        barrier["operands"] = [{"type": "IMM_UINT64", "text": "0x0", "bytes": 8}]
        g = capture(
            self.root,
            [[(0, 0, 0, 0, 1), (1, 0, 0, 0, 1), (2, 0, 0, 1, 1)]],
            [
                instruction(0, "STS", space="SHARED", write=True),
                barrier,
                instruction(2, "LDS", space="SHARED", read=True),
            ],
            block=2,
        )
        self.assertFalse(any(e["kind"] == "memory" for e in g["edges"]))

    def test_vector_load_replaces_all_four_registers(self):
        from .extract import register_contract

        inst = instruction(1, "LDS.128", space="SHARED", read=True)
        inst["size"] = 16
        inst["operands"] = [{"type": "REG", "text": "R8", "bytes": 16}, {"type": "MREF", "text": "[R4]", "bytes": 16}]
        writes, reads, known = register_contract(inst)
        self.assertEqual(writes, ["R8", "R9", "R10", "R11"])
        self.assertTrue(known)
        self.assertEqual(reads, [("R4", 1)])

    def test_alias_has_separate_output_value(self):
        capture(self.root, [[(0, 100, 0, 0, 1)]], [instruction(0, "STG", write=True)])
        g = build_graph(
            self.root / "trace", {}, [{"role": "input:0", "aliases": ["output:0"], "base": 100, "bytes": 4}]
        )
        self.assertTrue(any(n["kind"] == "output" for n in g["nodes"]))
        self.assertTrue(any(e["target"] == "result:output:0" and e["kind"] == "output_memory" for e in g["edges"]))

    def test_partial_capture_never_certifies_retention(self):
        from . import match

        g = capture(self.root, [[(0, 200, 0, 0, 1)]], [instruction(0, "STG", write=True)], dropped=1)
        g["context"] = {"task": "t", "model": "m", "input_signature": "i", "environment_signature": "e"}
        r = match.align_graphs(g, g)
        self.assertFalse(r["certifies_retention"])
        self.assertNotIn("l0:i0", [p["left"] for p in r["local_candidates"]])

    def test_uniform_carry_input_is_not_destination(self):
        from .extract import register_contract

        inst = instruction(0, "UIADD3.X", space="NONE")
        inst["operands"] = [
            {"type": kind, "text": text, "bytes": 4 if kind == "UREG" else 1}
            for kind, text in [
                ("UREG", "UR9"),
                ("UREG", "URZ"),
                ("UREG", "UR9"),
                ("UREG", "URZ"),
                ("UPRED", "UP0"),
                ("UPRED", "!UPT"),
            ]
        ]
        writes, reads, known = register_contract(inst)
        self.assertTrue(known)
        self.assertEqual(writes, ["UR9"])
        self.assertIn(("UP0", 4), reads)

    def test_uniform_carry_output_is_written(self):
        from .extract import register_contract

        inst = instruction(0, "UIADD3", space="NONE")
        inst["operands"] = [
            {"type": kind, "text": text, "bytes": 4 if kind == "UREG" else 1}
            for kind, text in [
                ("UREG", "UR8"),
                ("UPRED", "UP0"),
                ("UREG", "UR8"),
                ("IMM_UINT64", "0x40"),
                ("UREG", "URZ"),
            ]
        ]
        writes, reads, known = register_contract(inst)
        self.assertTrue(known)
        self.assertEqual(writes, ["UR8", "UP0"])


class WinnerContracts(unittest.TestCase):
    def test_no_correct_has_no_membership(self):
        r = trace_best([{"turn": 1, "correct": False, "score": 0}], None)
        self.assertEqual(r["status"], "no_correct_answer")

    def test_best_missing_not_replaced(self):
        r = trace_best(
            [{"turn": 1, "correct": True, "score": 1, "graph": {}}, {"turn": 2, "correct": True, "score": 2}], None
        )
        self.assertEqual(r["status"], "best_runtime_unavailable")
        self.assertEqual(r["best_turn"], 2)

    def test_tie_earliest(self):
        r = trace_best([{"turn": 2, "correct": True, "score": 1}, {"turn": 1, "correct": True, "score": 1}], None)
        self.assertEqual(r["best_turn"], 1)

    def test_invalid_score_rejected(self):
        with self.assertRaises(ValueError):
            trace_best([{"turn": 1, "correct": True, "score": float("nan")}], None)

    def test_duplicate_turns_rejected(self):
        with self.assertRaises(ValueError):
            trace_best([{"turn": 1}, {"turn": 1}], None)


class IndependentMatchingContracts(unittest.TestCase):
    def setUp(self):
        from . import match

        self.matcher = match

    def graph(self, labels, edges, unknown=()):
        return {
            "schema_version": "runtime_program_graph/v1",
            "id": "check",
            "context": {"task": "t", "model": "m", "input_signature": "i", "environment_signature": "e"},
            "nodes": [
                {
                    "id": str(i),
                    "kind": "instruction",
                    "attrs": {"op": op},
                    "unknowns": ["gap"] if i in unknown else [],
                    "evidence": {},
                }
                for i, op in enumerate(labels)
            ],
            "edges": [
                {"source": str(a), "target": str(b), "kind": "register", "attrs": {"port": p}, "unknowns": []}
                for a, b, p in edges
            ],
            "coverage": {"complete": False, "scope": "check", "unknowns": ["limited"]},
        }

    def test_renamed_ids_preserve_correspondence(self):
        a = self.graph(["load", "add", "store"], [(0, 1, 0), (1, 2, 0)])
        b = copy.deepcopy(a)
        for n in b["nodes"]:
            n["id"] = "other" + n["id"]
            n["evidence"] = {"source_name": "entirely different"}
        for e in b["edges"]:
            e["source"] = "other" + e["source"]
            e["target"] = "other" + e["target"]
        self.assertEqual(len(self.matcher.align_graphs(a, b)["local_candidates"]), 3)

    def test_changed_port_breaks_affected_match(self):
        a = self.graph(["load", "add", "store"], [(0, 1, 0), (1, 2, 0)])
        b = copy.deepcopy(a)
        b["edges"][0]["attrs"]["port"] = 1
        self.assertNotIn("1", [p["left"] for p in self.matcher.align_graphs(a, b)["local_candidates"]])

    def test_repeated_fragments_are_ambiguous(self):
        a = self.graph(["load", "add", "load", "add"], [(0, 1, 0), (2, 3, 0)])
        r = self.matcher.align_graphs(a, copy.deepcopy(a))
        self.assertFalse(r["local_candidates"])
        self.assertTrue(r["ambiguous"])

    def test_unknown_alternative_blocks_uniqueness(self):
        a = self.graph(["load", "add"], [(0, 1, 0)])
        b = self.graph(["load", "add", "load", "add"], [(0, 1, 0), (2, 3, 0)], unknown=(3,))
        self.assertFalse(self.matcher.align_graphs(a, b)["local_candidates"])

    def test_wrong_input_cannot_match(self):
        a = self.graph(["a", "b"], [(0, 1, 0)])
        b = copy.deepcopy(a)
        b["context"]["input_signature"] = "other"
        self.assertFalse(self.matcher.align_graphs(a, b)["local_candidates"])

    def test_budget_never_returns_unverified_match(self):
        a = self.graph(["a", "b"], [(0, 1, 0)])
        self.assertFalse(self.matcher.align_graphs(a, a, max_pairs=0)["local_candidates"])

    def test_earliest_copy_and_rollback(self):
        a = self.graph(["load", "add", "store"], [(0, 1, 0), (1, 2, 0)])
        b = self.graph(["load", "mul", "store"], [(0, 1, 0), (1, 2, 0)])
        r = trace_best(
            [
                {"turn": 1, "correct": True, "score": 1, "graph": a},
                {"turn": 2, "correct": True, "score": 1, "graph": b},
                {"turn": 3, "correct": True, "score": 2, "graph": a},
            ],
            self.matcher,
        )
        self.assertEqual({n["earliest_observed_correspondence_turn"] for n in r["candidates"]}, {1})

    def test_runtime_missing_does_not_mean_novel(self):
        a = self.graph(["load", "add", "store"], [(0, 1, 0), (1, 2, 0)])
        r = trace_best(
            [{"turn": 1, "correct": False}, {"turn": 2, "correct": True, "score": 1, "graph": a}], self.matcher
        )
        self.assertEqual({n["status"] for n in r["candidates"]}, {"origin_unknown"})
        self.assertEqual(r["missing_snapshots"], [{"turn": 1, "reason": "runtime_unavailable"}])


if __name__ == "__main__":
    unittest.main()
