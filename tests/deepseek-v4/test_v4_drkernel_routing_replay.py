"""Pins the drkernel custom generate's V4 routing-replay wiring.

V4's --use-rollout-routing-replay requires rollout_routed_experts on EVERY
sample; the default slime rollout captures it (return_routed_experts payload +
_decode_routed_experts) but the drkernel custom multi-turn generate did not →
"ValueError: rollout_routed_experts is required" in fill_routing_replay
(2026-07-04, drkernel smoke). Wired following upstream THUDM/slime 1b73ddc1's
shape contract (rows = len(tokens)-1). Source-level guards."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = (REPO / "examples" / "kernel_agent" / "generate_with_cuda_agent.py").read_text()


def test_payload_requests_routed_experts_when_replay_enabled():
    assert '"return_routed_experts"' in SRC, "payload must request routed experts"
    # gated on the replay flag, not unconditional
    payload_idx = SRC.index('"return_routed_experts"')
    gate_idx = SRC.rindex("use_rollout_routing_replay", 0, payload_idx)
    assert payload_idx - gate_idx < 400, "return_routed_experts must be gated on use_rollout_routing_replay"


def test_turn_sample_decodes_routed_experts():
    assert "_decode_routed_experts(" in SRC
    assert (
        "token_count=len(turn_sample.tokens) - 1" in SRC
    ), "row count must follow the upstream contract: len(tokens)-1"


def test_pad_turns_get_empty_routed_array():
    assert "fake_sample.rollout_routed_experts = routed[:0]" in SRC, (
        "pad turns must carry an empty (0-row) routed array so " "fill_routing_replay's per-sample invariant holds"
    )
