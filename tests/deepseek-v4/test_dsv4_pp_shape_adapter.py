"""Unit tests for the V4 pipeline-parallel tensor-shape adapter.

Megatron's non-interleaved PP schedule communicates ``[S, B, H]`` hidden
states; V4 stages pass the 4-stream hyper-connection hidden
``[B, S, hc_mult, H]``. ``_v4_pp_adjust_tensor_shapes_fn`` supplies the PP
p2p recv/send shapes for that stream (padded to the bshd data pad size so
non-first stages receive exactly what ``get_batch`` produces). Previously
proven only inside multi-node PP smokes.
"""

from argparse import Namespace
from types import SimpleNamespace

import pytest

from slime.backends.megatron_utils.model import _v4_pp_adjust_tensor_shapes_fn, _v4_pp_sequence_length


class _FakeV4Module:
    def __init__(self, hc_mult=4, hidden_size=4096):
        self.hf_config = SimpleNamespace(hc_mult=hc_mult, hidden_size=hidden_size)


def _args(**overrides):
    base = dict(
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=None,
        micro_batch_size=1,
        seq_length=64,
        qkv_format="bshd",
        tensor_model_parallel_size=1,
        data_pad_size_multiplier=1,
    )
    base.update(overrides)
    return Namespace(**base)


def test_adapter_returns_4stream_shape():
    fn = _v4_pp_adjust_tensor_shapes_fn(_args(), [_FakeV4Module()])
    assert fn is not None
    recv, send = fn(["ignored"], ["ignored"])
    assert recv == [(1, 64, 4, 4096)]
    assert send == [(1, 64, 4, 4096)]


def test_adapter_pads_sequence_to_bshd_pad_size():
    # seq 60 with TP2 * pad-multiplier 8 => pad to 64; PP p2p must match get_batch's padding
    args = _args(seq_length=60, tensor_model_parallel_size=2, data_pad_size_multiplier=8)
    assert _v4_pp_sequence_length(args) == 64
    fn = _v4_pp_adjust_tensor_shapes_fn(args, [_FakeV4Module()])
    recv, _ = fn([], [])
    assert recv == [(1, 64, 4, 4096)]


def test_adapter_no_padding_for_thd():
    args = _args(seq_length=60, qkv_format="thd", tensor_model_parallel_size=2, data_pad_size_multiplier=8)
    assert _v4_pp_sequence_length(args) == 60


def test_adapter_inactive_without_pp():
    assert _v4_pp_adjust_tensor_shapes_fn(_args(pipeline_model_parallel_size=1), [_FakeV4Module()]) is None


def test_adapter_inactive_for_non_v4_model():
    class _PlainModule:
        pass

    assert _v4_pp_adjust_tensor_shapes_fn(_args(), [_PlainModule()]) is None


def test_adapter_rejects_virtual_pp():
    args = _args(virtual_pipeline_model_parallel_size=2)
    with pytest.raises(ValueError, match="virtual pipeline"):
        _v4_pp_adjust_tensor_shapes_fn(args, [_FakeV4Module()])


def test_both_forward_paths_wire_the_adapter():
    """Regression guard: the 4D hc-stream PP adapter must be installed on BOTH
    the training forward AND the log-prob/forward_only path. Omitting it on the
    forward_only path made stage-1's compute_log_prob receive a 3D buffer and the
    mHC kernel crashed with 'not enough values to unpack (expected 4, got 3)'
    (Gate-A RL smoke, 2026-07-03). Source-level check so the exact regression
    can't silently return."""
    from pathlib import Path

    lines = (
        (Path(__file__).resolve().parents[2] / "slime" / "backends" / "megatron_utils" / "model.py")
        .read_text()
        .splitlines()
    )
    # Each forward_backward_func(...) call sets forward_only=... on its own line;
    # the matching adjust_tensor_shapes_fn=... must appear within the same call
    # (the very next few kwarg lines).
    forward_only_lines = [i for i, ln in enumerate(lines) if "forward_only=" in ln]
    assert len(forward_only_lines) >= 2, f"expected >=2 forward_only calls, got {len(forward_only_lines)}"
    for i in forward_only_lines:
        window = "\n".join(lines[i - 8 : i + 4])
        assert "adjust_tensor_shapes_fn=" in window, (
            "a forward_backward_func forward_only call is missing "
            "adjust_tensor_shapes_fn (V4 PP>1 hc-stream shape adapter) near line "
            f"{i + 1}"
        )


def test_adapter_uses_current_rollout_seq_len_override():
    """The adapter must size the PP hidden stream to the CURRENT rollout's padded
    max_seq_len (set on args._v4_pp_current_seq_len by the actor), not the static
    args.seq_length. Otherwise the RL log-prob forward communicates a stale short
    buffer while RoPE uses the real length (q S=128 vs cos/sin S=384 crash,
    Gate-A 2026-07-03)."""
    from slime.backends.megatron_utils.model import _v4_current_pp_seq_length

    args = _args(seq_length=64)
    assert _v4_current_pp_seq_length(args) == 64  # falls back when unset
    args._v4_pp_current_seq_len = 384
    assert _v4_current_pp_seq_length(args) == 384  # override wins

    fn = _v4_pp_adjust_tensor_shapes_fn(args, [_FakeV4Module(hc_mult=4, hidden_size=4096)])
    recv, send = fn([(64, 1, 4096)], [(64, 1, 4096)])
    # S dim must reflect the 384 override, not the 64 seq_length
    assert recv[0][1] == 384 and send[0][1] == 384, (recv, send)
    assert recv[0][2] == 4 and recv[0][3] == 4096  # hc_mult, hidden preserved
