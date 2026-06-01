import torch
from scripts.quantize.patches.llmcompressor_qwen3_5_smoothquant import smooth_weight_for_module
from scripts.quantize.producers.smoothquant_w8a8 import build_mappings_for_target, quant_targets_and_ignore
from torch import nn


class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + self.weight)


class PlainNorm(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())


def test_qwen3_5_rmsnorm_uses_offset_aware_smoothquant_formula():
    module = Qwen3_5RMSNorm(torch.tensor([-0.25, 0.0, 0.5, 1.0]))
    scales = torch.tensor([0.5, 1.0, 2.0, 4.0])

    smoothed = smooth_weight_for_module(module, scales)

    expected = (1.0 + module.weight) / scales - 1.0
    wrong_llmcompressor_formula = module.weight / scales
    assert torch.allclose(smoothed, expected)
    assert not torch.allclose(smoothed, wrong_llmcompressor_formula)


def test_plain_norm_keeps_llmcompressor_smoothquant_formula():
    module = PlainNorm(torch.tensor([-0.25, 0.0, 0.5, 1.0]))
    scales = torch.tensor([0.5, 1.0, 2.0, 4.0])

    assert torch.allclose(smooth_weight_for_module(module, scales), module.weight / scales)


def test_qwen3_5_rmsnorm_smoothquant_is_equivalent_before_linear():
    torch.manual_seed(0)
    module = Qwen3_5RMSNorm(torch.randn(4))
    linear = nn.Linear(4, 3, bias=False)
    x = torch.randn(5, 4)
    scales = torch.tensor([0.5, 1.25, 2.0, 4.0])

    reference = linear(module(x))

    smoothed_module = Qwen3_5RMSNorm(smooth_weight_for_module(module, scales).detach())
    smoothed_linear = nn.Linear(4, 3, bias=False)
    smoothed_linear.weight.data.copy_(linear.weight * scales.view(1, -1))

    assert torch.allclose(smoothed_linear(smoothed_module(x)), reference, atol=1e-6)


def test_smoothquant_mtp_targets_include_draft_head_mappings(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"text_config":{"layer_types":["linear_attention","full_attention"]}}\n')

    mlp_mtp = build_mappings_for_target("mlp_mtp", str(model_dir))
    nonla_mtp = build_mappings_for_target("non_linear_attn_mtp", str(model_dir))

    assert len(mlp_mtp) == 2
    assert any("mtp\\.layers" in pattern for pattern in mlp_mtp[-1][0])
    assert len(nonla_mtp) == 4
    assert any("self_attn" in pattern and "mtp\\.layers" in pattern for pattern in nonla_mtp[-1][0])


def test_smoothquant_mtp_targets_do_not_ignore_draft_scope():
    _, mlp_ignore = quant_targets_and_ignore("mlp")
    _, mlp_mtp_ignore = quant_targets_and_ignore("mlp_mtp")
    _, nonla_mtp_ignore = quant_targets_and_ignore("non_linear_attn_mtp")

    assert "re:.*mtp\\..*" in mlp_ignore
    assert "re:.*mtp\\..*" not in mlp_mtp_ignore
    assert "re:.*mtp\\..*" not in nonla_mtp_ignore
    assert r"re:^mtp\.fc(\..*)?$" in mlp_mtp_ignore
    assert r"re:^mtp\.fc(\..*)?$" in nonla_mtp_ignore
