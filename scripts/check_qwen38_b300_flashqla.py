"""Compare packed BF16 FlashQLA forward/backward against FLA at TP4/CP1 local head counts."""

import json

import torch
import torch.nn.functional as functional
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as reference_rule
from flash_qla import chunk_gated_delta_rule


def main():
    torch.manual_seed(1234)
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("This validation targets B300 SM103")
    results = []
    for sequence_lengths in ([256, 256], [4096, 512]):
        num_tokens = sum(sequence_lengths)
        query = torch.randn(1, num_tokens, 4, 128, device="cuda", dtype=torch.bfloat16)
        key = torch.randn_like(query)
        value = torch.randn(1, num_tokens, 12, 128, device="cuda", dtype=torch.bfloat16)
        gate = -torch.rand(1, num_tokens, 12, device="cuda") * 0.1
        beta = torch.sigmoid(torch.randn_like(gate)).to(torch.bfloat16)
        inputs = [functional.normalize(query, dim=-1), functional.normalize(key, dim=-1), value, gate, beta]
        inputs = [tensor.detach().requires_grad_() for tensor in inputs]
        reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
        boundaries = torch.tensor([0, sequence_lengths[0], num_tokens], device="cuda", dtype=torch.int32)
        expanded_inputs = [inputs[0].repeat_interleave(3, dim=2), inputs[1].repeat_interleave(3, dim=2), *inputs[2:]]
        expanded_reference = [
            reference_inputs[0].repeat_interleave(3, dim=2),
            reference_inputs[1].repeat_interleave(3, dim=2),
            *reference_inputs[2:],
        ]
        output, _ = chunk_gated_delta_rule(*expanded_inputs, cu_seqlens=boundaries, head_first=False)
        reference, _ = reference_rule(*expanded_reference, cu_seqlens=boundaries)
        gradient = torch.randn_like(output)
        output.backward(gradient)
        reference.backward(gradient)
        tensors = [("output", output, reference)] + [
            (name, tensor.grad, expected.grad)
            for name, tensor, expected in zip(
                ("query_grad", "key_grad", "value_grad", "gate_grad", "beta_grad"),
                inputs,
                reference_inputs,
                strict=True,
            )
        ]
        errors = {}
        for name, actual, expected in tensors:
            assert actual is not None and torch.isfinite(actual).all(), name
            assert expected is not None and torch.isfinite(expected).all(), name
            error = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-8)
            errors[name] = error.item()
            assert error < 0.03, f"{name}: relative L2 error {error.item()}"
        torch.cuda.synchronize()
        results.append({"sequence_lengths": sequence_lengths, "relative_l2_errors": errors})
        print(json.dumps(results[-1]), flush=True)
    print("PASS: native SM103 FlashQLA BF16 forward/backward agrees with FLA", flush=True)


if __name__ == "__main__":
    main()
