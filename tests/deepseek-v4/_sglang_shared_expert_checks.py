"""Serving-fork contract checks for shared-expert LoRA, run in a CLEAN
interpreter (no tilelang: its libcudart_stub shadows real cudart symbols that
sglang's imports resolve via ctypes, so importing sglang after the train-side
kernels flakes). Invoked by test_dsv4_lora_shared_expert.py via subprocess; also
runnable standalone on a serving container:

    python tests/deepseek-v4/_sglang_shared_expert_checks.py
"""

from __future__ import annotations

import sys
import types


def main() -> int:
    import torch  # noqa: F401  (before sglang; harmless in a clean process)

    # -- 1. target-token normalization: native w1/w3/w2 leaves ---------------
    from sglang.srt.lora.utils import get_normalized_target_modules

    got = get_normalized_target_modules(["w1", "w2", "w3", "wq_a"])
    assert {"gate_up_proj", "down_proj", "wq_a"} <= got, got
    print("PASS normalize_w123")

    # -- 2. weight renaming covers shared-expert paths ------------------------
    from sglang.srt.lora.lora import LoRAAdapter

    weights = {
        "base_model.model.layers.3.ffn.shared_experts.w1.lora_A.weight": torch.zeros(4, 8),
        "base_model.model.layers.3.ffn.shared_experts.w3.lora_A.weight": torch.zeros(4, 8),
        "base_model.model.layers.3.ffn.shared_experts.w2.lora_B.weight": torch.zeros(8, 4),
    }
    LoRAAdapter._rename_expert_w_to_proj(None, weights)
    assert "base_model.model.layers.3.ffn.shared_experts.gate_proj.lora_A.weight" in weights
    assert "base_model.model.layers.3.ffn.shared_experts.up_proj.lora_A.weight" in weights
    assert "base_model.model.layers.3.ffn.shared_experts.down_proj.lora_B.weight" in weights
    print("PASS rename_w_to_proj")

    # -- 3. V4 model LoRA metadata -------------------------------------------
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

    assert DeepseekV4ForCausalLM.lora_skip_fused_moe is True
    assert {"gate_up_proj", "down_proj"} <= set(DeepseekV4ForCausalLM.supported_lora_modules)
    assert set(DeepseekV4ForCausalLM.lora_replicated_modules) == {"gate_up_proj", "down_proj"}
    fake = types.SimpleNamespace(
        config=types.SimpleNamespace(hidden_size=4096, moe_intermediate_size=2048, n_shared_experts=1, head_dim=512)
    )
    # stock convention: gate_up returns the FULL stacked output dim
    assert DeepseekV4ForCausalLM.get_hidden_dim(fake, "gate_up_proj", 0) == (4096, 2 * 2048)
    assert DeepseekV4ForCausalLM.get_hidden_dim(fake, "down_proj", 0) == (2048, 4096)
    print("PASS v4_model_metadata")

    # -- 4. tp1 merged-column slice is identity for any global tp_rank --------
    from sglang.srt.lora.layers import MergedColumnParallelLinearWithLoRA

    B = torch.arange(12.0).reshape(12, 1)  # stacked gate(6)+up(6), r=1
    fake_layer = types.SimpleNamespace(
        base_layer=types.SimpleNamespace(tp_size=1, output_partition_sizes=[6, 6], output_sizes=[6, 6])
    )
    for tp_rank in range(8):
        out = MergedColumnParallelLinearWithLoRA.slice_lora_b_weights(fake_layer, B, tp_rank)
        assert torch.equal(out, B), f"tp_rank={tp_rank} must be identity for tp1 layer"
    print("PASS merged_slice_tp1")

    # -- 5. mem-pool sizes model-declared replicated modules FULL -------------
    from sglang.srt.lora.mem_pool import LoRAMemoryPool

    base_model = types.SimpleNamespace(
        lora_replicated_modules=("gate_up_proj", "down_proj"),
        get_hidden_dim=lambda name, idx: {
            "gate_up_proj": (4096, 2 * 2048),
            "down_proj": (2048, 4096),
        }[name],
    )
    fake_pool = types.SimpleNamespace(
        tp_size=8,
        moe_tp_size=1,
        max_loras_per_batch=2,
        base_hf_config=None,  # unused: base_model defines get_hidden_dim
        is_moe_module=lambda name: False,
    )
    assert LoRAMemoryPool.get_lora_A_shape(fake_pool, "gate_up_proj", base_model, 16, 0) == (2, 32, 4096)
    assert LoRAMemoryPool.get_lora_B_shape(fake_pool, "gate_up_proj", base_model, 16, 0) == (2, 4096, 16)
    assert LoRAMemoryPool.get_lora_A_shape(fake_pool, "down_proj", base_model, 16, 0) == (2, 16, 2048)
    assert LoRAMemoryPool.get_lora_B_shape(fake_pool, "down_proj", base_model, 16, 0) == (2, 4096, 16)
    print("PASS mem_pool_replicated_sizing")

    # -- 6. FusedMoE exclusion + waterfill/LoRA hard raise ---------------------
    import inspect

    from sglang.srt.lora.lora_manager import LoRAManager

    assert "lora_skip_fused_moe" in inspect.getsource(LoRAManager.init_lora_modules)

    # The mem pool's init_buffers treats gate_up/down as "ambiguous" when the
    # model has FusedMoE and allocates 4D ``*_moe`` per-expert buffers — that
    # path must also respect lora_skip_fused_moe (it calls
    # get_hidden_dim("gate_up_proj_moe"), which V4 does not implement; found
    # live by the truncated repro, scheduler died at init).
    import sglang.srt.lora.mem_pool as mem_pool_mod

    assert "lora_skip_fused_moe" in inspect.getsource(mem_pool_mod.LoRAMemoryPool.init_buffers)

    import sglang.srt.models.deepseek_v4 as v4mod

    orig = v4mod.get_global_server_args
    try:
        v4mod.get_global_server_args = lambda: types.SimpleNamespace(
            disable_shared_experts_fusion=False,
            enable_deepep_waterfill=True,
            enable_lora=True,
        )
        fake_model = types.SimpleNamespace(config=types.SimpleNamespace(n_shared_experts=1))
        try:
            v4mod.DeepseekV4ForCausalLM.determine_num_fused_shared_experts(fake_model)
            raise AssertionError("waterfill+lora must raise")
        except ValueError as e:
            assert "aterfill" in str(e)
    finally:
        v4mod.get_global_server_args = orig
    print("PASS fused_moe_skip_and_waterfill_raise")

    print("ALL SGLANG SHARED-EXPERT CONTRACTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
