import _cp_dist_helpers  # noqa: F401
import pytest
import torch

from megatron.core import mpu
from slime.backends.megatron_utils.loss import _build_topp_keep_mask, _fill_topp_mask_rows

NUM_GPUS = 0


def _set_cp(monkeypatch, *, size: int, rank: int) -> None:
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: size)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: rank)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)


def _kept_ids(row: torch.Tensor) -> list[int]:
    return row.nonzero(as_tuple=False).squeeze(-1).tolist()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (0, {2: [107]}),
        (1, {1: [104], 2: [105], 3: [106]}),
    ],
)
def test_top_p_mask_aligns_with_zigzag_cp_response_rows(monkeypatch, rank, expected):
    _set_cp(monkeypatch, size=2, rank=rank)
    keep = _build_topp_keep_mask(
        4,
        200,
        torch.device("cpu"),
        top_p_token_ids=[[104, 105, 106, 107]],
        top_p_token_offsets=[[0, 1, 2, 3, 4]],
        total_lengths=[8],
        response_lengths=[4],
        allgather_cp=False,
    )

    masked_rows = {row: _kept_ids(keep[row]) for row in range(keep.size(0)) if not keep[row].all()}
    assert masked_rows == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (0, {1: [102], 2: [103]}),
        (1, {0: [104], 1: [105]}),
    ],
)
def test_top_p_mask_aligns_with_allgather_cp_response_rows(monkeypatch, rank, expected):
    _set_cp(monkeypatch, size=2, rank=rank)
    keep = _build_topp_keep_mask(
        3,
        200,
        torch.device("cpu"),
        top_p_token_ids=[[102, 103, 104, 105]],
        top_p_token_offsets=[[0, 1, 2, 3, 4]],
        total_lengths=[6],
        response_lengths=[4],
        allgather_cp=True,
    )

    masked_rows = {row: _kept_ids(keep[row]) for row in range(keep.size(0)) if not keep[row].all()}
    assert masked_rows == expected


@pytest.mark.unit
def test_top_p_mask_aligns_with_cp1_response_rows(monkeypatch):
    _set_cp(monkeypatch, size=1, rank=0)
    keep = _build_topp_keep_mask(
        9,
        30,
        torch.device("cpu"),
        top_p_token_ids=[[13, 99, 14], [21, 22, 99, 23]],
        top_p_token_offsets=[[0, 2, 3], [0, 1, 3, 4]],
        total_lengths=[5, 4],
        response_lengths=[2, 3],
        allgather_cp=False,
    )

    masked_rows = {row: _kept_ids(keep[row]) for row in range(keep.size(0)) if not keep[row].all()}
    assert masked_rows == {2: [13], 3: [14], 5: [21], 6: [22], 7: [23]}


@pytest.mark.unit
@pytest.mark.parametrize("vocab_start", [0, 16, 32, 48])
@pytest.mark.parametrize("response_start,length", [(0, 12), (3, 5), (9, 20), (12, 0)])
def test_batched_top_p_mask_matches_rowwise_reference(vocab_start, response_start, length):
    generator = torch.Generator().manual_seed(71)
    sizes = torch.randint(0, 15, (12,), generator=generator)
    offsets = [0, *sizes.cumsum(0).tolist()]
    ids = torch.randint(0, 64, (offsets[-1],), generator=generator).tolist()
    expected = torch.ones(24, 16, dtype=torch.bool)
    actual = expected.clone()
    local_start = 2
    for response_index in range(response_start, min(response_start + length, 12)):
        row = local_start + response_index - response_start
        expected[row] = False
        selected = [
            token - vocab_start
            for token in ids[offsets[response_index] : offsets[response_index + 1]]
            if vocab_start <= token < vocab_start + 16
        ]
        expected[row, selected] = True
    _fill_topp_mask_rows(actual, ids, offsets, response_start, local_start, length, vocab_start, vocab_start + 16)
    assert torch.equal(actual, expected)


@pytest.mark.unit
def test_top_p_mask_keeps_missing_rows_but_clears_empty_nucleus(monkeypatch):
    _set_cp(monkeypatch, size=1, rank=0)
    actual = _build_topp_keep_mask(
        6,
        16,
        torch.device("cpu"),
        top_p_token_ids=[torch.tensor([3, 3, 20], dtype=torch.int32)],
        top_p_token_offsets=[torch.tensor([0, 3, 3], dtype=torch.int32)],
        total_lengths=[6],
        response_lengths=[3],
        allgather_cp=False,
    )
    assert _kept_ids(actual[2]) == [3]
    assert not actual[3].any()
    assert actual[4].all()


@pytest.mark.unit
@pytest.mark.parametrize(
    "cp_size,cp_rank,allgather_cp", [(1, 0, False), (2, 0, False), (2, 1, False), (2, 0, True), (2, 1, True)]
)
@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("chunk_size", [1, 3, 7])
def test_top_p_mask_row_intervals_match_full_mask(monkeypatch, cp_size, cp_rank, allgather_cp, tp_rank, chunk_size):
    _set_cp(monkeypatch, size=cp_size, rank=cp_rank)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda: tp_rank)
    total_rows = 16 // cp_size
    arguments = dict(
        T=total_rows,
        vocab_local=16,
        device=torch.device("cpu"),
        top_p_token_ids=[[3, 18, 7, 28], [4, 21]],
        top_p_token_offsets=[[0, 2, 2, 3, 4], [0, 1, 2]],
        total_lengths=[8, 8],
        response_lengths=[4, 3],
        allgather_cp=allgather_cp,
    )
    expected = _build_topp_keep_mask(**arguments)
    pieces = [
        _build_topp_keep_mask(**arguments, row_start=start, row_end=min(start + chunk_size, total_rows))
        for start in range(0, total_rows, chunk_size)
    ]
    assert all(piece.size(0) <= chunk_size for piece in pieces)
    assert torch.equal(torch.cat(pieces), expected)
    assert _build_topp_keep_mask(**arguments, row_start=total_rows, row_end=total_rows).shape == (0, 16)


@pytest.mark.unit
@pytest.mark.parametrize("row_start,row_end", [(-1, 2), (3, 2), (0, 5)])
def test_top_p_mask_rejects_invalid_row_interval(row_start, row_end):
    with pytest.raises(ValueError, match="Invalid top-p mask interval"):
        _build_topp_keep_mask(4, 16, torch.device("cpu"), [], [], [], [], False, row_start=row_start, row_end=row_end)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
