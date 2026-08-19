import base64

import numpy as np
import pytest


def _meta(values: np.ndarray) -> dict:
    return {"routed_experts": base64.b64encode(values.astype(np.int32).tobytes()).decode("ascii")}


def test_decode_routed_experts_infers_payload_topk_when_args_default_differs():
    from slime.rollout.sglang_rollout import _decode_routed_experts

    values = np.arange(28 * 43 * 6, dtype=np.int32)

    routed = _decode_routed_experts(
        _meta(values),
        token_count=28,
        num_layers=43,
        expected_topk=2,
    )

    assert routed.shape == (28, 43, 6)
    assert routed.dtype == np.int32
    np.testing.assert_array_equal(routed.reshape(-1), values)


def test_decode_routed_experts_rejects_non_integral_payload_shape():
    from slime.rollout.sglang_rollout import _decode_routed_experts

    with pytest.raises(ValueError, match="not divisible"):
        _decode_routed_experts(
            _meta(np.arange(28 * 43 * 6 + 1, dtype=np.int32)),
            token_count=28,
            num_layers=43,
            expected_topk=6,
        )
