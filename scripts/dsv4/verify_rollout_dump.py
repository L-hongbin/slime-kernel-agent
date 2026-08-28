#!/usr/bin/env python3
import argparse

import numpy as np
import torch


def verify(
    path: str,
    expected_samples: int,
    expected_layers: int,
    expected_topk: int,
    expected_num_experts: int = 256,
    expected_hash_layers: int = 3,
) -> None:
    data = torch.load(path, map_location="cpu", weights_only=False)
    samples = data.get("samples") or []
    print(f"debug_rollout_path={path}")
    print(f"num_samples={len(samples)}")
    assert len(samples) == expected_samples, len(samples)

    for i, sample in enumerate(samples):
        routed = sample.get("rollout_routed_experts")
        assert routed is not None, f"sample {i} missing rollout_routed_experts"
        arr = np.asarray(routed)
        print(
            f"sample{i}: status={sample.get('status')} response_len={sample.get('response_length')} "
            f"routed_shape={arr.shape} response={sample.get('response')!r}"
        )
        assert arr.ndim == 3, arr.shape
        assert arr.shape[0] > 0, f"sample {i} has empty routed token dimension"
        assert arr.shape[1] == expected_layers, arr.shape
        assert arr.shape[2] == expected_topk, arr.shape
        assert np.issubdtype(arr.dtype, np.integer), f"sample {i} routed dtype must be integer, got {arr.dtype}"
        assert int(arr.min()) >= 0, f"sample {i} routed expert id below 0"
        assert (
            int(arr.max()) < expected_num_experts
        ), f"sample {i} routed expert id {int(arr.max())} >= expected_num_experts={expected_num_experts}"
        assert (
            0 <= expected_hash_layers < expected_layers
        ), f"expected_hash_layers={expected_hash_layers} must be in [0, {expected_layers})"
        if expected_hash_layers:
            hash_layers = arr[:, :expected_hash_layers, :]
            learned_layers = arr[:, expected_hash_layers:, :]
            assert np.all(
                hash_layers == 0
            ), f"sample {i} expected hash layers 0..{expected_hash_layers - 1} to be all zero"
            assert np.any(learned_layers != 0), (
                f"sample {i} expected learned MoE layers {expected_hash_layers}..{expected_layers - 1} "
                "to include nonzero expert ids"
            )
            all_zero_layers = [layer for layer in range(expected_layers) if np.all(arr[:, layer, :] == 0)]
            print(
                f"sample{i}: all_zero_layers={all_zero_layers} "
                f"nonzero_layer_count={expected_layers - len(all_zero_layers)}"
            )

        tokens = sample.get("tokens")
        assert isinstance(tokens, list), f"sample {i} missing tokens list"
        assert (
            len(tokens) == arr.shape[0] + 1
        ), f"sample {i} expected len(tokens)=routed_tokens+1, got {len(tokens)} vs {arr.shape[0]}"

        response_len = sample.get("response_length")
        assert (
            isinstance(response_len, int) and response_len > 0
        ), f"sample {i} response_length must be a positive int, got {response_len!r}"
        assert response_len <= arr.shape[0], f"sample {i} response_length exceeds routed token count"
        assert isinstance(sample.get("status"), str) and sample["status"], f"sample {i} missing status"
        assert isinstance(sample.get("response"), str) and sample["response"], f"sample {i} missing response text"

    print("ROLLOUT_SMOKE_PASS routed_experts_present=true")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify R4 rollout smoke debug dump.")
    parser.add_argument("path")
    parser.add_argument("--expected-samples", type=int, default=2)
    parser.add_argument("--expected-layers", type=int, default=43)
    parser.add_argument("--expected-topk", type=int, default=6)
    parser.add_argument("--expected-num-experts", type=int, default=256)
    parser.add_argument("--expected-hash-layers", type=int, default=3)
    args = parser.parse_args()

    verify(
        args.path,
        args.expected_samples,
        args.expected_layers,
        args.expected_topk,
        args.expected_num_experts,
        args.expected_hash_layers,
    )


if __name__ == "__main__":
    main()
