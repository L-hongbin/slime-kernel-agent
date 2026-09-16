"""Focused executable checks for the offline best-answer membership heuristic."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from .best_answer_credit import select_winner, summarize_membership, trace_tokens

NUM_GPUS = 0


def turn(idx, speed=1.0, *, correct=True, complete=True):
    return {
        "turn_idx": idx,
        "observation": {
            "correctness": correct,
            "compiled": True,
            "decoy": False,
            "speedup": speed,
            "error_code": None,
        },
        "complete_sections": complete,
        "valid_modelnew": complete,
        "sections": {"CUDA_KERNELS": {"syntax_valid": True}},
        "components": [],
    }


def version(tokens, owners, idx=0):
    return {"tokens": tokens, "owners": owners, "turn": idx}


class CreditChecks(unittest.TestCase):
    @staticmethod
    def component_turn(idx, tokens, *, complete=True, section_valid=True):
        result = turn(idx, idx + 1, complete=complete)
        result["sections"]["CUDA_KERNELS"].update(source="source", syntax_valid=section_valid)
        result["components"] = [
            {
                "id": "kernel",
                "lineage": "c0",
                "section": "CUDA_KERNELS",
                "qualified_name": "kernel",
                "kind": "kernel",
                "syntax_hash": "/".join(tokens),
                "line": 1,
                "end_line": 2,
                "valid_syntax": True,
            }
        ]
        result["_test_tokens"] = tokens
        return result

    def membership(self, turns, coverage="component"):
        with patch(
            "tools.data.trajectory_structure.best_answer_credit.parsed_tokens",
            side_effect=lambda t: {"kernel": t["_test_tokens"]},
        ):
            return summarize_membership({"identity": ["test"], "turns": turns}, coverage=coverage)

    def test_valid_cuda_part_survives_missing_python_program(self):
        turns = [self.component_turn(0, ["code"], complete=False), self.component_turn(1, ["code"])]
        component = self.membership(turns)
        program = self.membership(turns, coverage="program")
        self.assertEqual(component["components"][0]["origin_token_counts"], {0: 1})
        self.assertEqual(program["components"][0]["unknown_tokens"], 1)

    def test_valid_local_node_survives_another_native_node_parse_error(self):
        turns = [self.component_turn(0, ["code"], section_valid=False), self.component_turn(1, ["code"])]
        self.assertEqual(self.membership(turns)["components"][0]["origin_token_counts"], {0: 1})

    def test_copy_turn_not_possible_origin_of_unmatched_tokens(self):
        turns = [
            self.component_turn(0, ["a", "old"]),
            self.component_turn(1, ["a", "new"]),
            self.component_turn(2, ["a", "new"]),
        ]
        result = self.membership(turns)
        self.assertGreater(result["components"][0]["unknown_tokens"], 0)
        self.assertEqual(result["turn_membership"][2]["membership"], 0)
        self.assertTrue(result["turn_membership"][2]["is_winner"])

    def test_exact_copy_preserves_origin(self):
        owners, method = trace_tokens(["a", "b"], [version(["a", "b"], [0, 0])], 2, gap=False)
        self.assertEqual(owners, [0, 0])
        self.assertEqual(method["method"], "exact_historical_version")

    def test_initial_snapshot_has_first_turn_origins(self):
        self.assertEqual(trace_tokens(["a", "b"], [], 0, gap=False)[0], [0, 0])

    def test_missing_prefix_cannot_prove_origin(self):
        self.assertEqual(trace_tokens(["a", "b"], [], 2, gap=True)[0], [None, None])

    def test_exact_revert_does_not_mint_new_origin(self):
        history = [version(["a", "b"], [0, 0]), version(["a", "c"], [0, 1], 1)]
        self.assertEqual(trace_tokens(["a", "b"], history, 2, gap=False)[0], [0, 0])

    def test_partial_revert_recovers_old_long_block(self):
        block = [str(x) for x in range(12)]
        h = [version([*block, "old"], [0] * 13), version(["replaced"], [1], 1)]
        owners, _ = trace_tokens([*block, "new"], h, 2, gap=False)
        self.assertEqual(owners, [0] * 12 + [2])

    def test_short_matches_remain_unknown(self):
        self.assertEqual(trace_tokens(["a", "new"], [version(["a", "old"], [0, 0])], 1, gap=False)[0], [None, 1])

    def test_repeated_text_is_not_unique(self):
        block = list("abcdefgh")
        h = [version(block + ["separator"] + block, [0] * 17)]
        owners, _ = trace_tokens(block + ["new"], h, 1, gap=False)
        self.assertEqual(owners[:8], [None] * 8)

    def test_code_gap_preserves_known_content_but_new_origin_unknown(self):
        block = list("abcdefgh")
        h = [version(block, [0] * 8)]
        owners, _ = trace_tokens(block + ["new"], h, 3, gap=True)
        self.assertEqual(owners, [0] * 8 + [None])

    def test_maximum_tie_chooses_earliest(self):
        self.assertEqual(select_winner([turn(3, 2), turn(1, 2), turn(0, 1)])["turn_idx"], 1)

    def test_no_valid_winner(self):
        self.assertIsNone(select_winner([turn(0, correct=False), turn(1, float("nan"))]))

    def test_bool_score_not_a_number(self):
        self.assertIsNone(select_winner([turn(0, True)]))

    def test_real_q_overrides_raw_speedup(self):
        self.assertEqual(select_winner([turn(0, 10), turn(1, 20)], scores={0: 2, 1: 1})["turn_idx"], 0)

    def test_legality_rejection_overrides_correct_flag(self):
        a = turn(0, 10)
        a["observation"]["decoy"] = True
        self.assertIsNone(select_winner([a]))

    def test_best_missing_source_does_not_silently_pick_worse(self):
        result = summarize_membership({"identity": ["test"], "turns": [turn(0, 1), turn(1, 2, complete=False)]})
        self.assertEqual(result["status"], "winner_structure_unavailable")
        self.assertEqual(result["winner_turn"], 1)

    def test_all_failure_returns_no_labels_not_zero_rewards(self):
        result = summarize_membership({"identity": ["test"], "turns": [turn(0, correct=False)]})
        self.assertEqual(result["status"], "no_valid_correct_winner")
        self.assertEqual(result["turn_membership"], [])

    def test_duplicate_turns_are_rejected(self):
        with self.assertRaises(ValueError):
            summarize_membership({"identity": ["test"], "turns": [turn(0), turn(0)]})

    def test_no_credit_after_winner(self):
        with patch("tools.data.trajectory_structure.best_answer_credit.parsed_tokens", return_value={}):
            result = summarize_membership({"identity": ["test"], "turns": [turn(0, 2), turn(1, 1)]})
        self.assertEqual(result["turn_membership"][1]["membership"], 0)

    def test_token_limit_is_unknown(self):
        owners, _ = trace_tokens(["a", "b", "c"], [version(["old"], [0])], 1, gap=False, token_limit=2)
        self.assertEqual(owners, [None, None, None])


if __name__ == "__main__":
    unittest.main()
