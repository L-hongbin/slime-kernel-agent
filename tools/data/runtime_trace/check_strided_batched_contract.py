"""CPU checks for generic cuBLAS strided-batched GEMM access contracts."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.data.runtime_trace.extract import build_graph, gemm_accesses


class StridedBatchedGemmContracts(unittest.TestCase):
    def attrs(self, **changes):
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
            "c": 1000,
            "lda": 2,
            "ldb": 4,
            "ldc": 2,
            "at": 0,
            "bt": 0,
            "ct": 0,
            "stride_a": 6,
            "stride_b": 12,
            "stride_c": 8,
            "batch_count": 3,
            **changes,
        }

    def test_per_batch_element_strides_become_byte_offsets(self):
        accesses = gemm_accesses(self.attrs())
        self.assertEqual(
            accesses,
            [
                ("a", "read", [[100, 172]]),
                ("b", "read", [[200, 344]]),
                ("c", "write", [[1000, 1096]]),
            ],
        )

    def test_stride_zero_reuses_one_operand_and_zero_alpha_skips_ab(self):
        accesses = gemm_accesses(self.attrs(alpha=0, stride_a=0, stride_b=0))
        self.assertEqual(
            accesses,
            [
                ("c", "write", [[1000, 1096]]),
            ],
        )

    def test_invalid_or_unbounded_batch_metadata_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "stride"):
            gemm_accesses(self.attrs(stride_a=-1))
        with self.assertRaisesRegex(ValueError, "batch count"):
            gemm_accesses(self.attrs(batch_count=100001))
        with self.assertRaisesRegex(ValueError, "interval budget"):
            gemm_accesses(
                self.attrs(n=50001, tb=0, ldb=4, stride_b=200004, c=10000000, stride_c=200000, batch_count=2)
            )

    def test_c_batches_and_c_operand_aliases_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "overlapping.*C batches"):
            gemm_accesses(self.attrs(stride_c=7))
        with self.assertRaisesRegex(ValueError, "C aliases A"):
            gemm_accesses(self.attrs(c=100))
        with self.assertRaisesRegex(ValueError, "C aliases B"):
            gemm_accesses(self.attrs(c=300))

    def test_a_b_broadcast_is_allowed_when_c_is_disjoint(self):
        accesses = gemm_accesses(self.attrs(stride_a=0, stride_b=0, c=1000))
        self.assertEqual([port for port, _, _ in accesses], ["a", "b", "c"])

    def test_ordinary_gemm_c_aliases_read_operands_fail_closed(self):
        regular = self.attrs()
        for key in ("stride_a", "stride_b", "stride_c", "batch_count"):
            regular.pop(key)
        with self.assertRaisesRegex(ValueError, "GEMM C aliases A"):
            gemm_accesses(regular | {"c": 100})
        with self.assertRaisesRegex(ValueError, "GEMM C aliases B"):
            gemm_accesses(regular | {"c": 200})
        # alpha=0 leaves A/B unread, so the ordinary C scaling path is valid.
        self.assertEqual(gemm_accesses(regular | {"alpha": 0, "c": 100}), [("c", "write", [(100, 132)])])

    def test_contract_maps_batched_operands_without_private_or_unmapped_gap(self):
        attrs = self.attrs(a=1000, b=2000, c=3000)
        records = [
            {"type": "config", "schema": "coarse-memory-runs/v1", "cta_limit": -1},
            {"type": "scope", "seq": 0, "enabled": True},
            {"type": "library_begin", "seq": 1, "api": "cublasSgemmStridedBatched", "stream": 0, "attrs": attrs},
            {"type": "library_end", "seq": 2, "parent": 1, "status": 0},
            {"type": "scope", "seq": 3, "enabled": False},
            {"type": "end", "seq": 4, "launches": 0},
        ]
        allocations = [
            {"id": key, "base": base, "bytes": 512, "first_seq": 0, "last_seq": None, "roles": []}
            for key, base in (("a", 1000), ("b", 2000), ("c", 3000))
        ]
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "trace"
            Path(str(prefix) + ".jsonl").write_text("\n".join(json.dumps(record) for record in records))
            graph = build_graph(prefix, allocations)
        node = graph["nodes"][0]
        self.assertTrue(node["footprint_complete"])
        self.assertNotIn("unmapped_contract_operand", node["unknowns"])
        self.assertEqual([access["buffer"] for access in node["reads"]], ["a", "b"])
        self.assertEqual([access["buffer"] for access in node["writes"]], ["c"])
        self.assertEqual(len(graph["versions"]), 1)


if __name__ == "__main__":
    unittest.main()
