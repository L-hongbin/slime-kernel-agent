import argparse
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from slime.utils.arguments import _resolve_checkpoint_load_args

NUM_GPUS = 0


def load_arguments_module(monkeypatch):
    megatron_mod = types.ModuleType("megatron")
    training_mod = types.ModuleType("megatron.training")
    arguments_mod = types.ModuleType("megatron.training.arguments")
    tokenizer_pkg_mod = types.ModuleType("megatron.training.tokenizer")
    tokenizer_mod = types.ModuleType("megatron.training.tokenizer.tokenizer")
    transformers_mod = types.ModuleType("transformers")

    arguments_mod.parse_args = lambda *args, **kwargs: None
    arguments_mod.validate_args = lambda args: args
    tokenizer_mod._vocab_size_with_padding = lambda vocab_size, _args: vocab_size
    transformers_mod.AutoConfig = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: None)

    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.training", training_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.arguments", arguments_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.tokenizer", tokenizer_pkg_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.tokenizer.tokenizer", tokenizer_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    module_path = Path(__file__).resolve().parents[1] / "slime" / "backends" / "megatron_utils" / "arguments.py"
    # Load inside the real package so relative imports such as
    # ``from .path_bootstrap import ...`` resolve against local structure.
    import slime.backends.megatron_utils  # noqa: F401

    module_name = "slime.backends.megatron_utils._arguments_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_slime_arguments_module(monkeypatch):
    router_pkg_mod = types.ModuleType("sglang_router")
    router_launch_mod = types.ModuleType("sglang_router.launch_router")
    sglang_arguments_mod = types.ModuleType("slime.backends.sglang_utils.arguments")
    sglang_external_mod = types.ModuleType("slime.backends.sglang_utils.external")
    logging_utils_mod = types.ModuleType("slime.observability.logging_utils")

    router_launch_mod.RouterArgs = object
    sglang_arguments_mod.sglang_parse_args = lambda *args, **kwargs: None
    sglang_arguments_mod.validate_args = lambda args: args
    sglang_external_mod.apply_external_engine_info_to_args = lambda *args, **kwargs: None
    logging_utils_mod.configure_logger = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "sglang_router", router_pkg_mod)
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", router_launch_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.arguments", sglang_arguments_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.external", sglang_external_mod)
    monkeypatch.setitem(sys.modules, "slime.observability.logging_utils", logging_utils_mod)

    module_path = Path(__file__).resolve().parents[1] / "slime" / "utils" / "arguments.py"
    module_name = "test_slime_argument_validation_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_qwen3_6_args(**overrides):
    values = dict(
        hidden_size=2048,
        num_attention_heads=16,
        num_layers=40,
        ffn_hidden_size=512,
        moe_ffn_hidden_size=512,
        moe_shared_expert_intermediate_size=512,
        moe_layer_freq=[1] * 40,
        untie_embeddings_and_output_weights=True,
        norm_epsilon=1e-6,
        layernorm_epsilon=1e-6,
        rotary_base=10000000,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def make_qwen3_6_hf_config():
    text_config = types.SimpleNamespace(
        hidden_size=2048,
        num_attention_heads=16,
        num_hidden_layers=40,
        intermediate_size=5632,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts=256,
        tie_word_embeddings=False,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 10000000},
    )
    return types.SimpleNamespace(text_config=text_config)


def make_allgather_cp_args(**overrides):
    values = dict(
        allgather_cp=True,
        context_parallel_size=2,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def make_default_megatron_args(**overrides):
    values = dict(
        optimizer="adam",
        use_distributed_optimizer=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
        fp16=False,
        seq_length=None,
        max_position_embeddings=None,
        dist_ckpt_save_pre_mcore_014=False,
        multi_latent_attention=False,
        vocab_size=None,
        padded_vocab_size=None,
        tokenizer_model=None,
        tokenizer_type=None,
        hf_checkpoint="/tmp/hf",
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


@pytest.mark.unit
def test_hf_validate_all_moe_skips_dense_intermediate_size(monkeypatch):
    module = load_arguments_module(monkeypatch)

    module._hf_validate_args(make_qwen3_6_args(), make_qwen3_6_hf_config())


@pytest.mark.unit
def test_hf_validate_checks_moe_intermediate_size(monkeypatch):
    module = load_arguments_module(monkeypatch)

    with pytest.raises(AssertionError, match="moe_intermediate_size"):
        module._hf_validate_args(make_qwen3_6_args(moe_ffn_hidden_size=256), make_qwen3_6_hf_config())


@pytest.mark.unit
def test_hf_validate_checks_dense_intermediate_size_when_moe_has_dense_layers(monkeypatch):
    module = load_arguments_module(monkeypatch)

    args = make_qwen3_6_args(moe_layer_freq=[0] + [1] * 39)

    with pytest.raises(AssertionError, match="intermediate_size"):
        module._hf_validate_args(args, make_qwen3_6_hf_config())


@pytest.mark.unit
def test_router_topk_comes_from_checkpoint(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = types.SimpleNamespace(moe_router_topk=2)
    hf_config = types.SimpleNamespace(model_type="deepseek_v4", num_experts_per_tok=6)

    module._bind_checkpoint_moe_router_topk(args, hf_config)

    assert args.moe_router_topk == 6


@pytest.mark.unit
def test_router_topk_rejects_explicit_checkpoint_conflict(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = types.SimpleNamespace(moe_router_topk=8)
    hf_config = types.SimpleNamespace(model_type="deepseek_v4", num_experts_per_tok=6)

    with pytest.raises(ValueError, match="conflicts with checkpoint metadata"):
        module._bind_checkpoint_moe_router_topk(args, hf_config, explicit_moe_router_topk=True)


@pytest.mark.unit
def test_non_ds_v4_router_topk_also_comes_from_checkpoint(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = types.SimpleNamespace(moe_router_topk=8)
    hf_config = types.SimpleNamespace(text_config=types.SimpleNamespace(model_type="qwen3_moe", num_experts_per_tok=6))

    module._bind_checkpoint_moe_router_topk(args, hf_config)

    assert args.moe_router_topk == 6


@pytest.mark.unit
def test_router_topk_ignores_checkpoint_without_topk_metadata(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = types.SimpleNamespace(moe_router_topk=8)
    hf_config = types.SimpleNamespace(model_type="dense")

    module._bind_checkpoint_moe_router_topk(args, hf_config)

    assert args.moe_router_topk == 8


@pytest.mark.unit
def test_allgather_cp_rejects_non_dsa_cp_models(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = make_allgather_cp_args()
    hf_config = types.SimpleNamespace(architectures=["Qwen3ForCausalLM"], model_type="qwen3")

    with pytest.raises(ValueError, match="only supported for DSA attention models"):
        module._validate_allgather_cp_supported(args, hf_config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "hf_config",
    [
        types.SimpleNamespace(architectures=["DeepseekV32ForCausalLM"], model_type="deepseek_v3"),
        types.SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"], model_type="glm"),
    ],
)
def test_allgather_cp_allows_dsa_architectures(monkeypatch, hf_config):
    module = load_arguments_module(monkeypatch)

    module._validate_allgather_cp_supported(make_allgather_cp_args(), hf_config)


@pytest.mark.unit
def test_allgather_cp_ignores_cp_size_one(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = make_allgather_cp_args(context_parallel_size=1)

    module._validate_allgather_cp_supported(args)


@pytest.mark.unit
def test_default_args_keep_distributed_optimizer_for_adam(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = make_default_megatron_args(optimizer="adam")

    module._set_default_megatron_args(args)

    assert args.use_distributed_optimizer is True
    assert args.bf16 is True


@pytest.mark.unit
def test_default_args_disable_distributed_optimizer_for_muon(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = make_default_megatron_args(
        optimizer="muon",
        use_distributed_optimizer=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        overlap_param_gather_with_optimizer_step=True,
    )

    module._set_default_megatron_args(args)

    assert args.use_distributed_optimizer is False
    assert args.overlap_grad_reduce is False
    assert args.overlap_param_gather is False
    assert args.overlap_param_gather_with_optimizer_step is False
    assert args.bf16 is True


@pytest.mark.unit
@pytest.mark.parametrize(("start_rollout_id", "expected"), [(100, 100), (None, 0)])
def test_checkpoint_fallback_preserves_explicit_start_rollout_id(start_rollout_id, expected):
    args = types.SimpleNamespace(
        load=None,
        ref_load=None,
        hf_checkpoint="/tmp/hf",
        ref_ckpt_step=7,
        ckpt_step=None,
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        start_rollout_id=start_rollout_id,
    )

    _resolve_checkpoint_load_args(args)

    assert args.start_rollout_id == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_tag", "expected_load_step"),
    [("ref", 12), ("teacher", 34), ("rollout_actor", 77)],
)
def test_load_other_checkpoint_restores_ckpt_step(monkeypatch, model_tag, expected_load_step):
    from slime.backends.megatron_utils import actor as actor_module

    actor = types.SimpleNamespace(
        args=types.SimpleNamespace(
            load="original-load",
            no_load_optim=False,
            no_load_rng=False,
            finetune=False,
            ckpt_step=77,
            ref_ckpt_step=12,
            opd_teacher_ckpt_step=34,
        ),
        model=object(),
        weights_backuper=types.SimpleNamespace(backup=lambda tag: None),
        _active_model_tag="actor",
    )
    seen = {}

    def fake_load_checkpoint(*args, **kwargs):
        seen["load"] = actor.args.load
        seen["ckpt_step"] = actor.args.ckpt_step
        return 0, 0

    monkeypatch.setattr(actor_module, "load_checkpoint", fake_load_checkpoint)

    actor_module.MegatronTrainRayActor.load_other_checkpoint(actor, model_tag, "/tmp/other-checkpoint")

    assert seen == {"load": "/tmp/other-checkpoint", "ckpt_step": expected_load_step}
    assert actor.args.load == "original-load"
    assert actor.args.no_load_optim is False
    assert actor.args.no_load_rng is False
    assert actor.args.finetune is False
    assert actor.args.ckpt_step == 77
    assert actor._active_model_tag == model_tag


@pytest.mark.unit
def test_update_weight_disk_dir_required_for_disk_transport(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(update_weight_transport="disk", update_weight_disk_dir=None)

    with pytest.raises(ValueError, match="update-weight-disk-dir"):
        module.slime_validate_args(args)


def make_slime_validate_args(**overrides):
    values = dict(
        eval_config=None,
        eval_prompt_data=None,
        kl_coef=0,
        use_kl_loss=False,
        ref_load=None,
        use_opd=False,
        opd_type=None,
        opd_teacher_load=None,
        load=None,
        hf_checkpoint="/tmp/hf",
        ref_ckpt_step=None,
        ckpt_step=None,
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        start_rollout_id=None,
        eval_interval=None,
        save_interval=None,
        save=None,
        kl_loss_coef=0,
        advantage_estimator="grpo",
        normalize_advantages=False,
        use_rollout_logprobs=False,
        use_tis=False,
        get_mismatch_metrics=False,
        custom_tis_function_path=None,
        use_dynamic_batch_size=False,
        max_tokens_per_gpu=None,
        log_probs_max_tokens_per_gpu=None,
        balance_by_flops=False,
        balance_data=False,
        eps_clip_high=None,
        eps_clip=0.2,
        eval_reward_key=None,
        reward_key="reward",
        dump_details=None,
        save_debug_rollout_data=None,
        save_debug_train_data=None,
        load_debug_rollout_data=None,
        rollout_external_engine_addrs=None,
        debug_train_only=False,
        actor_num_gpus_per_node=8,
        actor_num_nodes=1,
        offload=False,
        offload_train=None,
        offload_rollout=None,
        debug_rollout_only=False,
        colocate=False,
        rollout_num_gpus=8,
        eval_function_path=None,
        rollout_function_path="custom.rollout",
        num_steps_per_rollout=None,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        global_batch_size=None,
        grpo_std_normalization=True,
        over_sampling_batch_size=None,
        num_epoch=None,
        num_rollout=1,
        rollout_global_dataset=False,
        enable_mtp_training=False,
        mtp_num_layers=None,
        use_rollout_routing_replay=False,
        use_routing_replay=False,
        custom_config_path=None,
        eval_max_context_len=None,
        rollout_max_context_len=None,
        rollout_max_prompt_len=None,
        train_backend="megatron",
        release_train=False,
        keep_old_actor=False,
        only_train_params_name_list=None,
        freeze_params_name_list=None,
        update_weight_transport="nccl",
        update_weight_disk_dir=None,
        update_weight_local_checkpoint_dir=None,
        update_weight_mode="full",
        rollout_temperature=1.0,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


@pytest.mark.unit
def test_slime_validate_args_preserves_explicit_start_rollout_id(monkeypatch):
    """``--start-rollout-id`` is only a fallback when the user did not set it."""
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(start_rollout_id=100)

    module.slime_validate_args(args)

    assert args.start_rollout_id == 100


@pytest.mark.unit
def test_slime_validate_args_defaults_start_rollout_id_to_zero(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(start_rollout_id=None)

    module.slime_validate_args(args)

    assert args.start_rollout_id == 0


@pytest.mark.unit
def test_slime_validate_args_rejects_equal_debug_data_paths(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        save_debug_rollout_data="/tmp/debug_{rollout_id}.pt",
        save_debug_train_data="/tmp/debug_{rollout_id}.pt",
    )

    with pytest.raises(ValueError, match="--save-debug-train-data must not be equal"):
        module.slime_validate_args(args)


@pytest.mark.unit
@pytest.mark.parametrize("temperature", [0.0, -0.1])
def test_slime_validate_args_rejects_non_positive_rollout_temperature(monkeypatch, temperature):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(rollout_temperature=temperature)

    with pytest.raises(ValueError, match="--rollout-temperature must be > 0"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_slime_validate_args_preserves_zero_rollout_gpus_under_colocate(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(colocate=True, rollout_num_gpus=0)

    module.slime_validate_args(args)

    assert args.rollout_num_gpus == 0
    assert args.offload_train is True
    assert args.offload_rollout is True


@pytest.mark.unit
def test_slime_validate_args_preserves_larger_rollout_gpus_under_colocate(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        colocate=True,
        actor_num_gpus_per_node=8,
        actor_num_nodes=1,
        rollout_num_gpus=12,
    )

    module.slime_validate_args(args)

    assert args.rollout_num_gpus == 12
    assert args.offload_train is True
    assert args.offload_rollout is True


@pytest.mark.unit
def test_slime_validate_args_preserves_zero_rollout_gpus_without_colocate(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(colocate=False, rollout_num_gpus=0)

    module.slime_validate_args(args)

    assert args.rollout_num_gpus == 0
    assert args.actor_num_gpus_per_node == 8
    assert args.actor_num_nodes == 1
    assert args.offload_train is False
    assert args.offload_rollout is False


@pytest.mark.unit
def test_update_weight_delta_requires_disk_transport(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        update_weight_mode="delta",
        update_weight_transport="nccl",
        update_weight_local_checkpoint_dir="/local/ckpt",
    )

    with pytest.raises(ValueError, match="requires --update-weight-transport=disk"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_update_weight_delta_rejects_colocate(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        update_weight_mode="delta",
        update_weight_transport="disk",
        update_weight_disk_dir="/shared/delta",
        update_weight_local_checkpoint_dir="/local/ckpt",
        colocate=True,
    )

    with pytest.raises(ValueError, match="not supported with --colocate"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_update_weight_delta_requires_local_checkpoint_dir(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        update_weight_mode="delta",
        update_weight_transport="disk",
        update_weight_disk_dir="/shared/delta",
        update_weight_local_checkpoint_dir=None,
    )

    with pytest.raises(ValueError, match="requires --update-weight-local-checkpoint-dir"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_force_fp8_ue8m0_scale_argument(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    parser = argparse.ArgumentParser()
    module.get_slime_extra_args_provider()(parser)

    defaults = parser.parse_args(["--rollout-batch-size", "1"])
    configured = parser.parse_args(["--rollout-batch-size", "1", "--force-fp8-ue8m0-scale"])

    assert defaults.force_fp8_ue8m0_scale is False
    assert configured.force_fp8_ue8m0_scale is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
