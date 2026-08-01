from types import SimpleNamespace

import torch


def _router_cfg():
    return SimpleNamespace(
        num_experts_per_tok=2,
        num_local_experts=4,
        hidden_size=3,
        scoring_func="sigmoid",
        routed_scaling_factor=1.0,
        vocab_size=16,
    )


class _FakeReplay:
    def __init__(self):
        self.recorded = []

    def record(self, indices):
        self.recorded.append(indices.detach().clone())

    def pop_forward(self):
        return self.recorded[0]

    def pop_backward(self):
        return self.recorded[0]


def test_dsv4_topk_router_uses_routing_replay(monkeypatch):
    from custom_kernels.deepseek_v4.megatron.mcore_model import V4TopKRouter

    monkeypatch.setenv("ENABLE_ROUTING_REPLAY", "1")
    monkeypatch.setenv("ROUTING_REPLAY_STAGE", "record")
    router = V4TopKRouter(_router_cfg())
    router.routing_replay = _FakeReplay()

    with torch.no_grad():
        router.weight.copy_(
            torch.tensor(
                [
                    [3.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ]
            )
        )
        router.e_score_correction_bias.zero_()
    hidden = torch.tensor([[[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]]])

    _, _, recorded_indices = router(hidden)
    assert router.routing_replay.recorded

    with torch.no_grad():
        router.weight.copy_(
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                ]
            )
        )

    monkeypatch.setenv("ROUTING_REPLAY_STAGE", "fallthrough")
    _, _, natural_indices = router(hidden)
    assert not torch.equal(natural_indices, recorded_indices)

    monkeypatch.setenv("ROUTING_REPLAY_STAGE", "replay_forward")
    _, _, replayed_indices = router(hidden)
    assert torch.equal(replayed_indices, recorded_indices)


def test_dsv4_topk_registers_but_hash_router_does_not(monkeypatch):
    from custom_kernels.deepseek_v4.megatron.mcore_model import V4HashRouter, V4TopKRouter

    from slime.utils.routing_replay import RoutingReplay

    monkeypatch.setenv("ENABLE_ROUTING_REPLAY", "1")
    RoutingReplay.all_routing_replays.clear()

    topk = V4TopKRouter(_router_cfg())
    assert hasattr(topk, "routing_replay")
    assert len(RoutingReplay.all_routing_replays) == 1

    hash_router = V4HashRouter(_router_cfg())
    assert not hasattr(hash_router, "routing_replay")
    assert len(RoutingReplay.all_routing_replays) == 1

    RoutingReplay.all_routing_replays.clear()


def test_actor_routing_replay_skips_v4_hash_layers():
    from slime.utils.routing_replay import should_skip_rollout_routing_replay_layer

    model = SimpleNamespace(
        layer_ids=(0, 1, 2, 3),
        layers=[
            SimpleNamespace(mlp=SimpleNamespace(is_hash=True)),
            SimpleNamespace(mlp=SimpleNamespace(is_hash=True)),
            SimpleNamespace(mlp=SimpleNamespace(is_hash=True)),
            SimpleNamespace(mlp=SimpleNamespace(is_hash=False)),
        ],
    )

    assert should_skip_rollout_routing_replay_layer(model, 0) is True
    assert should_skip_rollout_routing_replay_layer(model, 2) is True
    assert should_skip_rollout_routing_replay_layer(model, 3) is False
    assert should_skip_rollout_routing_replay_layer(model, 99) is False
    assert should_skip_rollout_routing_replay_layer(SimpleNamespace(config=object()), 0) is False


def test_rollout_routing_replay_finds_real_layer_gate_through_wrappers():
    from slime.utils.routing_replay import (
        ROUTING_REPLAY_LAYER_NOT_FOUND,
        get_rollout_routing_replay_for_layer,
        should_skip_rollout_routing_replay_layer,
    )

    replay_1 = _FakeReplay()
    replay_3 = _FakeReplay()
    inner = SimpleNamespace(
        layer_ids=(0, 1, 2, 3),
        layers=[
            SimpleNamespace(mlp=SimpleNamespace(is_hash=True)),
            SimpleNamespace(mlp=SimpleNamespace(is_hash=False, gate=SimpleNamespace(routing_replay=replay_1))),
            SimpleNamespace(mlp=SimpleNamespace(is_hash=True)),
            SimpleNamespace(mlp=SimpleNamespace(is_hash=False, gate=SimpleNamespace(routing_replay=replay_3))),
        ],
    )
    wrapped = SimpleNamespace(module=SimpleNamespace(model=inner))

    assert get_rollout_routing_replay_for_layer(wrapped, 0) is None
    assert get_rollout_routing_replay_for_layer(wrapped, 1) is replay_1
    assert get_rollout_routing_replay_for_layer(wrapped, 2) is None
    assert get_rollout_routing_replay_for_layer(wrapped, 3) is replay_3
    assert get_rollout_routing_replay_for_layer(wrapped, 99) is ROUTING_REPLAY_LAYER_NOT_FOUND

    assert should_skip_rollout_routing_replay_layer(wrapped, 0) is True
    assert should_skip_rollout_routing_replay_layer(wrapped, 1) is False
    assert should_skip_rollout_routing_replay_layer(wrapped, 99) is False


def test_record_rollout_routing_replay_uses_layer_gate_not_global_offset():
    from slime.utils.routing_replay import RoutingReplay, record_rollout_routing_replay_for_layer

    replay_1 = _FakeReplay()
    replay_3 = _FakeReplay()
    wrong_0 = _FakeReplay()
    wrong_1 = _FakeReplay()
    RoutingReplay.all_routing_replays[:] = [wrong_0, wrong_1]
    try:
        model = SimpleNamespace(
            layer_ids=(0, 1, 2, 3),
            layers=[
                SimpleNamespace(mlp=SimpleNamespace(is_hash=True)),
                SimpleNamespace(
                    mlp=SimpleNamespace(
                        is_hash=False,
                        gate=SimpleNamespace(routing_replay=replay_1),
                    )
                ),
                SimpleNamespace(mlp=SimpleNamespace(is_hash=True)),
                SimpleNamespace(
                    mlp=SimpleNamespace(
                        is_hash=False,
                        gate=SimpleNamespace(routing_replay=replay_3),
                    )
                ),
            ],
        )
        routed = torch.arange(20, dtype=torch.long).reshape(5, 4)

        offset = 0
        for layer_id in range(4):
            offset = record_rollout_routing_replay_for_layer(
                model,
                layer_id,
                routed[:, layer_id],
                offset,
            )

        assert offset == 2
        torch.testing.assert_close(replay_1.recorded[0], routed[:, 1])
        torch.testing.assert_close(replay_3.recorded[0], routed[:, 3])
        assert wrong_0.recorded == []
        assert wrong_1.recorded == []
    finally:
        RoutingReplay.all_routing_replays[:] = []


def test_record_rollout_routing_replay_falls_back_to_global_offset_for_unknown_model():
    from slime.utils.routing_replay import RoutingReplay, record_rollout_routing_replay_for_layer

    fallback = _FakeReplay()
    RoutingReplay.all_routing_replays[:] = [fallback]
    try:
        routed = torch.arange(5, dtype=torch.long)
        offset = record_rollout_routing_replay_for_layer(
            SimpleNamespace(config=object()),
            7,
            routed,
            0,
        )

        assert offset == 1
        torch.testing.assert_close(fallback.recorded[0], routed)
    finally:
        RoutingReplay.all_routing_replays[:] = []


def test_fill_routing_replay_pads_bshd_to_max_seqlen():
    """Regression guard: in bshd, fill_routing_replay must pad each sample's
    recorded routing to the rollout-wide max_seq_len (matching get_batch), not
    just to pad_size. Otherwise a short sample yields too few routing rows vs the
    globally-padded [B, max_seq_len] tokens the router flattens -> "replayed V4
    top-k indices shape [128,6] does not match scores rows 256" (Gate-B scale,
    2026-07-03). Source-level check so the bshd branch can't silently regress to
    the pad_size-only path."""
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "slime" / "backends" / "megatron_utils" / "actor.py").read_text()
    # Isolate fill_routing_replay's body.
    m = re.search(r"def fill_routing_replay\(.*?\n(.*?)\n    def ", src, flags=re.S)
    body = m.group(1) if m else src
    assert 'qkv_format == "bshd"' in body, "fill_routing_replay must branch on bshd"
    # The bshd branch must pass qkv_format + a max-seqlen to slice_with_cp.
    assert re.search(r"slice_with_cp\(\s*r,\s*pad_func,\s*self\.args\.qkv_format,\s*max_seqlen", body), (
        "bshd fill_routing_replay must call slice_with_cp with qkv_format + max_seqlen "
        "so routing is padded to the rollout-wide max_seq_len"
    )
    assert (
        'rollout_data["max_seq_lens"][0]' in body
    ), "bshd max_seqlen must come from the same rollout-wide max_seq_lens get_batch uses"
