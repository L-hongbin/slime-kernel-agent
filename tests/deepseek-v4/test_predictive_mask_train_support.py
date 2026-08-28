"""CPU tests for predictive Top-K train-side support alignment and gathering."""

import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slime.backends.megatron_utils import cp_utils
from slime.backends.megatron_utils import loss as loss_module
from slime.backends.megatron_utils.cp_utils import CP_PARTITION_CONTIGUOUS, slice_log_prob_with_cp
from slime.backends.megatron_utils.loss import (
    _embed_cp_local_response_values,
    _extract_per_sample,
    _validate_dppo_predictive_support_batch,
    _vocab_parallel_selected_log_probs,
    get_dppo_predictive_support_log_probs,
)

NUM_GPUS = 0


def _patch_parallel(monkeypatch, *, cp_size=1, cp_rank=0, tp_size=1, tp_rank=0):
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: cp_size)
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_rank", lambda: cp_rank)
    monkeypatch.setattr(loss_module.mpu, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(loss_module.mpu, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(loss_module.mpu, "get_tensor_model_parallel_group", lambda: None)


def test_selected_log_probs_tp1_match_full_log_softmax():
    logits = torch.tensor(
        [[0.0, 1.0, -1.0, 2.0, 0.5], [2.0, -2.0, 0.0, 1.0, 3.0], [0.2, 0.1, 0.0, -0.1, -0.2]],
        dtype=torch.float32,
    )
    ids = torch.tensor([[3, 1, 0], [4, 0, 2], [2, 0, 0]], dtype=torch.long)
    valid = torch.tensor([[True, True, False], [True, True, True], [False, False, False]])
    actual = _vocab_parallel_selected_log_probs(
        logits,
        ids,
        valid,
        tp_group=None,
        tp_rank=0,
        tp_world_size=1,
        chunk_size=2,
    )
    expected = torch.log_softmax(logits, dim=-1).gather(1, ids)
    expected.masked_fill_(~valid, 0.0)
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_cp2_bshd_support_embedding_round_trips_local_order(monkeypatch, cp_rank):
    _patch_parallel(monkeypatch, cp_size=2, cp_rank=cp_rank)
    old_mode = cp_utils.get_cp_partition_mode()
    cp_utils.set_cp_partition_mode(CP_PARTITION_CONTIGUOUS)
    try:
        total_length, response_length, max_seq_len = 12, 6, 16
        full = torch.arange(response_length * 3, dtype=torch.long).view(response_length, 3)
        local = slice_log_prob_with_cp(full, total_length, response_length, "bshd", max_seq_len)
        embedded = _embed_cp_local_response_values(
            [local],
            logits_rows=max_seq_len // 2,
            total_lengths=[total_length],
            response_lengths=[response_length],
            qkv_format="bshd",
            max_seq_lens=[max_seq_len],
            allgather_cp=False,
        )
        extracted, _ = _extract_per_sample(
            embedded,
            None,
            [total_length],
            [response_length],
            "bshd",
            [max_seq_len],
            False,
        )
        torch.testing.assert_close(extracted[0], local)
    finally:
        cp_utils.set_cp_partition_mode(old_mode)


@pytest.mark.parametrize("temperature", [1.0, 1.4])
def test_current_support_log_probs_use_response_predictor_rows(monkeypatch, temperature):
    _patch_parallel(monkeypatch)
    args = Namespace(
        qkv_format="thd",
        allgather_cp=False,
        rollout_temperature=temperature,
        log_probs_chunk_size=2,
        vocab_size=5,
    )
    logits = torch.tensor(
        [
            [
                [3.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0, 0.0],
                [9.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    ids = torch.tensor([[4, 2, 0], [0, 3, 1]], dtype=torch.long)
    valid = torch.ones_like(ids, dtype=torch.bool)
    actual = get_dppo_predictive_support_log_probs(
        logits,
        args=args,
        support_token_ids=[ids],
        support_valid_masks=[valid],
        total_lengths=[4],
        response_lengths=[2],
        max_seq_lens=None,
    )[0]
    expected = torch.log_softmax(logits[0, 1:3] / temperature, dim=-1).gather(1, ids)
    torch.testing.assert_close(actual, expected)


def _valid_batch(*, second_row_active=True):
    sampled_probs = torch.tensor([0.2, 0.3], dtype=torch.float32)
    old = sampled_probs.log()
    ids = torch.tensor([[2, 0, 0], [3, 1, 0]], dtype=torch.long)
    logs = torch.tensor(
        [[old[0].item(), torch.tensor(0.1).log().item(), 0.0], [old[1].item(), torch.tensor(0.2).log().item(), 0.0]],
        dtype=torch.float32,
    )
    valid = torch.tensor([[True, True, False], [True, True, False]])
    batch = {
        "rollout_topk_token_ids": [ids],
        "rollout_topk_log_probs": [logs],
        "rollout_topk_valid_mask": [valid],
        "response_lengths": [2],
        "total_lengths": [4],
        "unconcat_tokens": [torch.tensor([4, 4, 2, 3], dtype=torch.long)],
        "loss_masks": [torch.tensor([1, int(second_row_active)], dtype=torch.int)],
    }
    return batch, old


def test_support_validation_checks_sampled_token_and_anchor(monkeypatch):
    _patch_parallel(monkeypatch)
    args = Namespace(dppo_predictive_top_k=2, qkv_format="thd", vocab_size=5)
    batch, old = _valid_batch()
    ids, logs, valid = _validate_dppo_predictive_support_batch(args, batch, old)
    assert ids.shape == logs.shape == valid.shape == (2, 3)

    batch["rollout_topk_token_ids"][0][1, 0] = 4
    with pytest.raises(ValueError, match="sampled token exactly once"):
        _validate_dppo_predictive_support_batch(args, batch, old)


def test_support_validation_allows_empty_padding_row(monkeypatch):
    _patch_parallel(monkeypatch)
    args = Namespace(dppo_predictive_top_k=2, qkv_format="thd", vocab_size=5)
    batch, old = _valid_batch(second_row_active=False)
    batch["rollout_topk_valid_mask"][0][1].fill_(False)
    _validate_dppo_predictive_support_batch(args, batch, old)


def test_support_validation_rejects_duplicate_ids(monkeypatch):
    _patch_parallel(monkeypatch)
    args = Namespace(dppo_predictive_top_k=2, qkv_format="thd", vocab_size=5)
    batch, old = _valid_batch()
    batch["rollout_topk_token_ids"][0][0, 1] = 2
    with pytest.raises(ValueError, match="duplicate token ids"):
        _validate_dppo_predictive_support_batch(args, batch, old)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
