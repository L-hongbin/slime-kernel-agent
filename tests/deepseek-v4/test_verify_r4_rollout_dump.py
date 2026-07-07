import numpy as np
import pytest
import torch

from scripts.v4.verify_rollout_dump import verify


def _routed(tokens=4):
    routed = np.zeros((tokens, 43, 6), dtype=np.int64)
    routed[:, 3:, :] = 1
    return routed


def _sample(routed, *, tokens=None, response_length=1):
    return {
        "status": "ok",
        "tokens": list(range(routed.shape[0] + 1)) if tokens is None else tokens,
        "response_length": response_length,
        "response": "x",
        "rollout_routed_experts": routed,
    }


def test_verify_r4_rollout_dump_accepts_expected_routing_shape(tmp_path, capsys):
    path = tmp_path / "rollout_0.pt"
    routed = _routed()
    torch.save(
        {
            "samples": [
                _sample(routed),
                _sample(routed),
            ]
        },
        path,
    )

    verify(str(path), expected_samples=2, expected_layers=43, expected_topk=6)

    output = capsys.readouterr().out
    assert "num_samples=2" in output
    assert "routed_shape=(4, 43, 6)" in output
    assert "all_zero_layers=[0, 1, 2]" in output
    assert "ROLLOUT_SMOKE_PASS routed_experts_present=true" in output


def test_verify_r4_rollout_dump_rejects_missing_routing(tmp_path):
    path = tmp_path / "rollout_0.pt"
    torch.save({"samples": [{"status": "ok"}, {"status": "ok"}]}, path)

    with pytest.raises(AssertionError, match="missing rollout_routed_experts"):
        verify(str(path), expected_samples=2, expected_layers=43, expected_topk=6)


def test_verify_r4_rollout_dump_rejects_bad_routing_values(tmp_path):
    path = tmp_path / "rollout_0.pt"
    routed = _routed().astype(np.int32)
    routed[0, 3, 0] = 256
    torch.save({"samples": [_sample(routed), _sample(_routed().astype(np.int32))]}, path)

    with pytest.raises(AssertionError, match="expected_num_experts=256"):
        verify(str(path), expected_samples=2, expected_layers=43, expected_topk=6)


def test_verify_r4_rollout_dump_rejects_non_integer_routing(tmp_path):
    path = tmp_path / "rollout_0.pt"
    routed = _routed().astype(np.float32)
    torch.save({"samples": [_sample(routed), _sample(routed)]}, path)

    with pytest.raises(AssertionError, match="dtype must be integer"):
        verify(str(path), expected_samples=2, expected_layers=43, expected_topk=6)


def test_verify_r4_rollout_dump_rejects_token_alignment_mismatch(tmp_path):
    path = tmp_path / "rollout_0.pt"
    routed = _routed().astype(np.int32)
    torch.save({"samples": [_sample(routed, tokens=[1, 2]), _sample(routed)]}, path)

    with pytest.raises(AssertionError, match=r"len\(tokens\)=routed_tokens\+1"):
        verify(str(path), expected_samples=2, expected_layers=43, expected_topk=6)


def test_verify_r4_rollout_dump_rejects_nonzero_hash_layers(tmp_path):
    path = tmp_path / "rollout_0.pt"
    routed = _routed()
    routed[0, 1, 0] = 7
    torch.save({"samples": [_sample(routed), _sample(_routed())]}, path)

    with pytest.raises(AssertionError, match="expected hash layers 0..2 to be all zero"):
        verify(str(path), expected_samples=2, expected_layers=43, expected_topk=6)


def test_verify_r4_rollout_dump_rejects_all_zero_learned_layers(tmp_path):
    path = tmp_path / "rollout_0.pt"
    routed = np.zeros((4, 43, 6), dtype=np.int64)
    torch.save({"samples": [_sample(routed), _sample(_routed())]}, path)

    with pytest.raises(AssertionError, match="expected learned MoE layers 3..42"):
        verify(str(path), expected_samples=2, expected_layers=43, expected_topk=6)
