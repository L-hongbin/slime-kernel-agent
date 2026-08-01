"""CPU unit tests for the CP2 (contiguous context-parallel) slime plumbing.

Covers the slime-side pieces of the DeepSeek-V4-Flash contiguous-CP design
(handoffs/deepseek-v4/cp2_design.md, findings B1/M3/M4):

  (a) ``slice_with_cp`` contiguous mode == the manual contiguous block slice,
      including the pad_func-callable (routing-replay) path.
  (b) ``get_logits_and_tokens_offset_with_cp`` contiguous mode: per-rank
      token/logit offsets exactly TILE the response with no gap/overlap
      (property test over random spans, incl. rank-boundary crossings), and
      each rank owns exactly ``[r*l_local, (r+1)*l_local)`` (M3).
  (c) B1 pad-granularity: ``compute_cp_padded_max_seq_len`` yields
      ``l_local % 128 == 0`` for CP2 across raw lengths 1..40k.
  (d) M4 MTP contiguous roll: a real 2/3-rank gloo group verifies the global
      left-shift-by-one equivalence (interior + boundary ranks).
  (e) ``cp_size == 1`` is a strict no-op for (a) and (b) in either mode.

The environment ships a real ``megatron.core.mpu``; tests monkeypatch its
``get_context_parallel_{rank,world_size}`` to stand in for a launched CP group
(the same object ``cp_utils`` bound at import time).
"""

from __future__ import annotations

import random
import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.backends.megatron_utils import cp_utils
from slime.backends.megatron_utils.cp_utils import (
    CP_PARTITION_CONTIGUOUS,
    CP_PARTITION_ZIGZAG,
    all_gather_with_cp,
    compute_cp_padded_max_seq_len,
    get_logits_and_tokens_offset_with_cp,
    slice_log_prob_with_cp,
    slice_with_cp,
)

NUM_GPUS = 0


def _set_cp(monkeypatch, cp_size: int, cp_rank: int) -> None:
    from megatron.core import mpu

    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: cp_size)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: cp_rank)


# ---------------------------------------------------------------------------
# (a) slice_with_cp — contiguous mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("cp_size,max_seq_len", [(2, 256), (2, 512), (4, 512)])
def test_slice_with_cp_contiguous_bshd(monkeypatch, cp_size, max_seq_len):
    """Each rank's contiguous slice == the manual [r*l_local, (r+1)*l_local) block."""
    l_local = max_seq_len // cp_size
    assert l_local % 128 == 0  # precondition B1 guarantees
    raw_len = max_seq_len - 37  # some real doc shorter than the padded width
    tokens = torch.arange(1, raw_len + 1, dtype=torch.long)
    padded = F.pad(tokens, (0, max_seq_len - raw_len), value=0)

    for r in range(cp_size):
        _set_cp(monkeypatch, cp_size, r)
        out = slice_with_cp(tokens, 0, "bshd", max_seq_len, partition_mode=CP_PARTITION_CONTIGUOUS)
        expected = padded[r * l_local : (r + 1) * l_local]
        assert out.shape[0] == l_local
        assert torch.equal(out, expected), f"rank {r}: {out} vs {expected}"


@pytest.mark.unit
def test_slice_with_cp_contiguous_padfunc_routing_replay(monkeypatch):
    """The routing-replay path passes a *callable* pad_value (a 3-D experts
    tensor [S, num_layers, topk]); contiguous slice must match the manual
    pad-then-block slice."""
    cp_size, max_seq_len = 2, 256
    l_local = max_seq_len // cp_size
    num_layers, topk, num_experts = 3, 2, 8
    raw_len = 130
    experts = torch.arange(raw_len * num_layers * topk, dtype=torch.long).reshape(raw_len, num_layers, topk)

    def pad_func(x, pad):
        _, nl, tk = x.shape
        block = torch.arange(pad * nl * tk, device=x.device, dtype=x.dtype).reshape(pad, nl, tk) % num_experts
        return torch.cat([x, block], dim=0)

    padded = pad_func(experts, max_seq_len - raw_len)

    for r in range(cp_size):
        _set_cp(monkeypatch, cp_size, r)
        out = slice_with_cp(experts, pad_func, "bshd", max_seq_len, partition_mode=CP_PARTITION_CONTIGUOUS)
        expected = padded[r * l_local : (r + 1) * l_local]
        assert out.shape == (l_local, num_layers, topk)
        assert torch.equal(out, expected), f"rank {r} routing slice mismatch"


@pytest.mark.unit
def test_slice_with_cp_contiguous_asserts_l_local_alignment(monkeypatch):
    """B1 guard: a bshd max_seq_len whose l_local is not 128-aligned must raise."""
    _set_cp(monkeypatch, 2, 0)
    # cp2, max_seq_len=200 -> chunk_size=ceil(200/4)=50 -> l_local=100, not %128.
    with pytest.raises(AssertionError, match="multiple of 128"):
        slice_with_cp(torch.arange(50), 0, "bshd", 200, partition_mode=CP_PARTITION_CONTIGUOUS)


# ---------------------------------------------------------------------------
# (b) get_logits_and_tokens_offset_with_cp — contiguous offsets tile the response
# ---------------------------------------------------------------------------


def _collect_intervals(monkeypatch, cp_size, total_length, response_length, max_seq_len, mode):
    """Gather every rank's (token, logit) offset intervals for one sample."""
    tok, log = [], []
    for r in range(cp_size):
        _set_cp(monkeypatch, cp_size, r)
        _, _, logits_off, tokens_off = get_logits_and_tokens_offset_with_cp(
            total_length, response_length, "bshd", max_seq_len, partition_mode=mode
        )
        for h in range(2):
            if tokens_off[h][1] > tokens_off[h][0]:
                tok.append(tuple(tokens_off[h]))
            if logits_off[h][1] > logits_off[h][0]:
                log.append(tuple(logits_off[h]))
    return tok, log


def _assert_tiles(intervals, lo, hi):
    """intervals (half-open) must partition [lo, hi) exactly: sorted, adjacent, no gap/overlap."""
    intervals = sorted(intervals)
    assert intervals, f"no intervals to tile [{lo},{hi})"
    assert intervals[0][0] == lo, f"first start {intervals[0][0]} != {lo}"
    assert intervals[-1][1] == hi, f"last end {intervals[-1][1]} != {hi}"
    for (s0, e0), (s1, e1) in zip(intervals, intervals[1:], strict=False):
        assert e0 == s1, f"gap/overlap between [{s0},{e0}) and [{s1},{e1})"


@pytest.mark.unit
@pytest.mark.parametrize("mode", [CP_PARTITION_CONTIGUOUS, CP_PARTITION_ZIGZAG])
def test_offset_tiles_response_property(monkeypatch, mode):
    """Random response spans (incl. rank-boundary crossings) must exactly tile
    the response in both token- and logit-space, for both partition modes."""
    rng = random.Random(20260718)
    cp_size = 2
    for _ in range(400):
        total_length = rng.randint(2, 4000)
        prompt_length = rng.randint(1, total_length - 1)  # >=1 prompt token
        response_length = total_length - prompt_length
        max_seq_len = compute_cp_padded_max_seq_len(total_length, 128, cp_size, mode)

        tok, log = _collect_intervals(monkeypatch, cp_size, total_length, response_length, max_seq_len, mode)
        # token space tiles [prompt_length, total_length); logit space [prompt-1, total-1).
        _assert_tiles(tok, prompt_length, total_length)
        _assert_tiles(log, prompt_length - 1, total_length - 1)


@pytest.mark.unit
def test_offset_tiles_deliberate_boundary_crossing(monkeypatch):
    """Force a span that straddles the CP2 rank boundary l_local."""
    cp_size = 2
    max_seq_len = 512  # l_local = 256
    l_local = max_seq_len // cp_size
    total_length = 400  # response reaches past l_local=256
    prompt_length = 100  # prompt entirely in rank0, response spans both ranks
    response_length = total_length - prompt_length
    tok, log = _collect_intervals(
        monkeypatch, cp_size, total_length, response_length, max_seq_len, CP_PARTITION_CONTIGUOUS
    )
    _assert_tiles(tok, prompt_length, total_length)
    _assert_tiles(log, prompt_length - 1, total_length - 1)
    # And the span really does cross: some tokens live in rank0's block [0,256),
    # some in rank1's block [256,512).
    assert any(s < l_local for s, _ in tok) and any(e > l_local for _, e in tok)


@pytest.mark.unit
@pytest.mark.parametrize("cp_size,max_seq_len", [(2, 512), (4, 1024)])
def test_contiguous_rank_owns_single_block(monkeypatch, cp_size, max_seq_len):
    """M3 core: contiguous rank r owns exactly the contiguous block
    [r*l_local, (r+1)*l_local) (union of its two adjacent chunks)."""
    l_local = max_seq_len // cp_size
    # A long response so both chunks clip to non-empty ranges everywhere.
    total_length = max_seq_len
    response_length = max_seq_len - 1
    for r in range(cp_size):
        _set_cp(monkeypatch, cp_size, r)
        chunk_size, chunks_off, _, _ = get_logits_and_tokens_offset_with_cp(
            total_length, response_length, "bshd", max_seq_len, partition_mode=CP_PARTITION_CONTIGUOUS
        )
        assert 2 * chunk_size == l_local
        # two adjacent chunks whose union is the rank's contiguous block
        assert chunks_off[0] == (r * l_local, r * l_local + chunk_size)
        assert chunks_off[1] == (r * l_local + chunk_size, (r + 1) * l_local)


# ---------------------------------------------------------------------------
#  slice_log_prob_with_cp through the module-global mode (internal-caller path)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_slice_log_prob_contiguous_reconstructs_response(monkeypatch):
    """slice_log_prob_with_cp reads the process-wide mode; under contiguous CP
    the per-rank slices concatenate (in rank order) back to the full response."""
    prev = cp_utils.get_cp_partition_mode()
    cp_utils.set_cp_partition_mode(CP_PARTITION_CONTIGUOUS)
    try:
        cp_size = 2
        total_length, response_length, max_seq_len = 300, 200, 512
        log_prob = torch.arange(response_length, dtype=torch.float32)
        pieces = []
        for r in range(cp_size):
            _set_cp(monkeypatch, cp_size, r)
            pieces.append(slice_log_prob_with_cp(log_prob, total_length, response_length, "bshd", max_seq_len))
        recon = torch.cat(pieces)
        assert torch.equal(recon, log_prob), f"{recon} != {log_prob}"
    finally:
        cp_utils.set_cp_partition_mode(prev)


@pytest.mark.unit
def test_bshd_actual_length_slice_gather_round_trip(monkeypatch):
    """Gather must reuse the padded BSHD width that produced each CP slice.

    This is the exact shape that first failed in the fixed-batch old-actor+TIS
    experiment: the physical width is 5120 while the sample's real length is
    only 4520.  Recomputing offsets from the real length expects 975 local rows
    on rank 0, although the BSHD forward correctly produced 1275.
    """
    prev = cp_utils.get_cp_partition_mode()
    cp_utils.set_cp_partition_mode(CP_PARTITION_CONTIGUOUS)
    try:
        cp_size = 2
        total_length, response_length, max_seq_len = 4520, 3234, 5120
        full = torch.arange(response_length, dtype=torch.float32)
        contributions = []
        monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_group", lambda: None)
        monkeypatch.setattr(cp_utils.dist.nn, "all_reduce", lambda tensor, group=None: tensor)
        for rank in range(cp_size):
            _set_cp(monkeypatch, cp_size, rank)
            local = slice_log_prob_with_cp(full, total_length, response_length, "bshd", max_seq_len)
            contributions.append(
                all_gather_with_cp(
                    local,
                    total_length,
                    response_length,
                    qkv_format="bshd",
                    max_seq_len=max_seq_len,
                )
            )

        assert contributions[0].count_nonzero() > 0
        assert contributions[1].count_nonzero() > 0
        assert torch.equal(sum(contributions), full)
    finally:
        cp_utils.set_cp_partition_mode(prev)


# ---------------------------------------------------------------------------
# (c) B1 pad-granularity: l_local % 128 == 0 across raw lengths
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("cp_size", [2, 4])
def test_pad_granularity_l_local_128_aligned(cp_size):
    rng = random.Random(7)
    base_pad = 1 * 128  # tp_size=1, data_pad_size_multiplier=128
    raw_lengths = [1, 2, 127, 128, 129, 255, 256, 257, 8192, 16384, 40000] + [
        rng.randint(1, 40000) for _ in range(500)
    ]
    for raw in raw_lengths:
        m = compute_cp_padded_max_seq_len(raw, base_pad, cp_size, CP_PARTITION_CONTIGUOUS)
        assert m >= raw
        assert m % (cp_size * 128) == 0, (raw, m, cp_size)
        assert (m // cp_size) % 128 == 0, (raw, m, cp_size)


@pytest.mark.unit
def test_pad_granularity_zigzag_and_cp1_unchanged():
    base_pad = 128
    # zigzag: B1 does not apply -> plain round-up to base_pad.
    assert compute_cp_padded_max_seq_len(300, base_pad, 2, CP_PARTITION_ZIGZAG) == 384
    # cp_size==1: no-op regardless of mode.
    assert compute_cp_padded_max_seq_len(300, base_pad, 1, CP_PARTITION_CONTIGUOUS) == 384


# ---------------------------------------------------------------------------
# (e) cp_size == 1 no-op identity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cp1_slice_with_cp_noop_identity(monkeypatch):
    _set_cp(monkeypatch, 1, 0)
    tokens = torch.arange(50, dtype=torch.long)

    # thd: pure passthrough, mode-independent.
    z = slice_with_cp(tokens, 0, "thd", partition_mode=CP_PARTITION_ZIGZAG)
    c = slice_with_cp(tokens, 0, "thd", partition_mode=CP_PARTITION_CONTIGUOUS)
    assert torch.equal(z, tokens) and torch.equal(c, tokens)

    # bshd: pad-to-max_seq_len, mode-independent.
    max_seq_len = 64
    expected = F.pad(tokens, (0, max_seq_len - 50), value=0)
    z = slice_with_cp(tokens, 0, "bshd", max_seq_len, partition_mode=CP_PARTITION_ZIGZAG)
    c = slice_with_cp(tokens, 0, "bshd", max_seq_len, partition_mode=CP_PARTITION_CONTIGUOUS)
    assert torch.equal(z, expected) and torch.equal(c, expected)


@pytest.mark.unit
def test_cp1_offset_helper_is_guarded_and_slice_log_prob_passthrough(monkeypatch):
    _set_cp(monkeypatch, 1, 0)
    # get_logits_and_tokens_offset_with_cp is cp>1 only (both modes).
    for mode in (CP_PARTITION_ZIGZAG, CP_PARTITION_CONTIGUOUS):
        with pytest.raises(AssertionError):
            get_logits_and_tokens_offset_with_cp(12, 8, "bshd", 64, partition_mode=mode)
    # slice_log_prob_with_cp returns the input unchanged at cp1 (mode-independent).
    lp = torch.arange(8, dtype=torch.float32)
    assert torch.equal(slice_log_prob_with_cp(lp, 12, 8, "bshd", 64), lp)


# ---------------------------------------------------------------------------
# (d) M4 MTP contiguous roll on a real gloo CP group
# ---------------------------------------------------------------------------


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _mtp_roll_worker(rank: int, world_size: int, port: int, l_local: int, batch: int):
    import os

    import torch as _torch
    import torch.distributed as _dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    _dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        cp_group = _dist.new_group(ranks=list(range(world_size)))
        from custom_kernels.deepseek_v4.megatron.mtp import roll_tensor_contiguous_cp

        seq = world_size * l_local
        # deterministic non-trivial global sequence [batch, seq]
        g = _torch.arange(1, batch * seq + 1, dtype=_torch.float32).reshape(batch, seq)
        local = g[:, rank * l_local : (rank + 1) * l_local].contiguous()

        rolled_local, s = roll_tensor_contiguous_cp(local, shifts=-1, dims=-1, cp_group=cp_group)
        assert _torch.isclose(s, rolled_local.sum()), "returned sum must match rolled tensor"

        def _gather(t):
            buf = [_torch.empty_like(t) for _ in range(world_size)]
            _dist.all_gather(buf, t.contiguous())
            return _torch.cat(buf, dim=-1)

        # single roll == global left-shift-by-one, last global slot zeroed.
        got1 = _gather(rolled_local)
        ref1 = _torch.roll(g, shifts=-1, dims=-1)
        ref1[:, -1] = 0
        assert _torch.equal(got1, ref1), f"rank{rank}: 1x roll {got1} != global roll {ref1}"

        # composing two rolls (the loss-side roll_mtp_labels_and_masks path) ==
        # global left-shift-by-two, last two global slots zeroed.
        rolled2_local, _ = roll_tensor_contiguous_cp(rolled_local, shifts=-1, dims=-1, cp_group=cp_group)
        got2 = _gather(rolled2_local)
        ref2 = _torch.roll(g, shifts=-2, dims=-1)
        ref2[:, -2:] = 0
        assert _torch.equal(got2, ref2), f"rank{rank}: 2x roll {got2} != global roll-by-2 {ref2}"
    finally:
        _dist.destroy_process_group()


@pytest.mark.unit
@pytest.mark.parametrize("world_size", [2, 3])
def test_mtp_contiguous_roll_matches_global_shift(world_size):
    """The contiguous-CP MTP roll (local torch.roll + one right-neighbour
    boundary exchange) reproduces a global left-shift-by-one. world_size=3
    exercises an interior rank (both sends and receives)."""
    if not torch.distributed.is_gloo_available():
        pytest.skip("gloo backend unavailable")
    port = _free_port()
    torch.multiprocessing.spawn(
        _mtp_roll_worker,
        args=(world_size, port, 4, 2),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.unit
def test_mtp_contiguous_roll_cp1_is_plain_roll():
    """cp_group=None (CP=1) is a plain left torch.roll with a zeroed last slot."""
    from custom_kernels.deepseek_v4.megatron.mtp import roll_tensor_contiguous_cp

    x = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)
    rolled, s = roll_tensor_contiguous_cp(x, shifts=-1, dims=-1, cp_group=None)
    ref = torch.roll(x, shifts=-1, dims=-1)
    ref[:, -1] = 0
    assert torch.equal(rolled, ref)
    assert torch.isclose(s, ref.sum())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
