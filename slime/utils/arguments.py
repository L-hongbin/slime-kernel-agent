import argparse
import copy
import json
import logging
import math
import os
from typing import Any

import yaml
from sglang_router.launch_router import RouterArgs

from slime.backends.sglang_utils.arguments import sglang_parse_args
from slime.backends.sglang_utils.arguments import validate_args as sglang_validate_args
from slime.utils.eval_config import EvalDatasetConfig, build_eval_dataset_configs, ensure_dataset_list
from slime.utils.logging_utils import configure_logger

logger = logging.getLogger(__name__)


def _validate_lora_args(args) -> None:
    dim = int(getattr(args, "lora_dim", 0) or 0)
    alpha = getattr(args, "lora_alpha", None)
    dropout = float(getattr(args, "lora_dropout", 0.0) or 0.0)
    plus_lambda = getattr(args, "lora_plus_lambda", None)
    max_node_bytes = int(getattr(args, "lora_checkpoint_max_node_bytes", 2 * 1024**3))

    if dim < 0:
        raise ValueError(f"--lora-dim must be non-negative, got {dim}")
    if dim > 0 and alpha is not None and int(alpha) <= 0:
        raise ValueError(f"--lora-alpha must be positive when LoRA is enabled, got {alpha}")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"--lora-dropout must be in [0, 1), got {dropout}")
    if plus_lambda is not None and float(plus_lambda) <= 0:
        raise ValueError(f"--lora-plus-lambda must be positive, got {plus_lambda}")
    if max_node_bytes <= 0:
        raise ValueError("--lora-checkpoint-max-node-bytes must be positive, " f"got {max_node_bytes}")
    if dim == 0 and (
        alpha is not None
        or dropout != 0.0
        or getattr(args, "lora_rslora", False)
        or plus_lambda not in (None, 1, 1.0)
        or getattr(args, "dsv4_lora_shared_expert", False)
        or getattr(args, "lora_adapter_resume_load", "")
        or getattr(args, "use_lora_weight_sync", False)
    ):
        raise ValueError("LoRA options require --lora-dim to be positive")


def _validate_dppo_predictive_args(args) -> None:
    """Validate that rollout and train distributions match for predictive Top-K KL."""
    if getattr(args, "policy_loss_mode", "ppo") != "dppo_topk_kl_predictive":
        return
    top_k = getattr(args, "dppo_predictive_top_k", 0)
    if top_k <= 0:
        raise ValueError("dppo_topk_kl_predictive requires --dppo-predictive-top-k to be a positive integer.")
    vocab_size = getattr(args, "vocab_size", None)
    if vocab_size is not None and top_k >= vocab_size:
        raise ValueError(f"--dppo-predictive-top-k ({top_k}) must be smaller than vocab_size ({vocab_size}).")
    if not args.use_rollout_logprobs:
        raise ValueError(
            "dppo_topk_kl_predictive requires --use-rollout-logprobs: the Top-K distribution "
            "and sampled-token importance ratio must share the rollout behavior-policy anchor."
        )
    if args.use_tis:
        raise ValueError("dppo_topk_kl_predictive is incompatible with --use-tis.")
    if args.eps_clip <= 0 or args.eps_clip_high != args.eps_clip:
        raise ValueError(
            "dppo_topk_kl_predictive uses one positive KL threshold delta; set "
            "--eps-clip and --eps-clip-high to the same positive value."
        )
    if getattr(args, "rollout_temperature", 1.0) != 1.0:
        logger.warning(
            "dppo_topk_kl_predictive is running with --rollout-temperature=%s instead of 1. "
            "The rollout sampled-token/Top-K log-probs and train-side current-policy "
            "probabilities must all come from the same temperature-scaled distribution.",
            args.rollout_temperature,
        )
    if getattr(args, "rollout_top_p", 1.0) != 1.0 or getattr(args, "rollout_top_k", -1) != -1:
        raise ValueError(
            "dppo_topk_kl_predictive currently requires --rollout-top-p 1 and "
            "--rollout-top-k -1 so rollout behavior probabilities and train probabilities "
            "refer to the same untruncated distribution."
        )
    if getattr(args, "allgather_cp", False):
        raise ValueError(
            "dppo_topk_kl_predictive does not yet support --allgather-cp; its contiguous-logit "
            "layout needs a separate Top-K support redistribution. Use the regular CP path."
        )


def _validate_partial_rollout_args(args) -> None:
    if getattr(args, "rollout_weight_sync_pause_mode", "abort") == "retract" and not getattr(
        args, "partial_rollout", False
    ):
        raise ValueError("--rollout-weight-sync-pause-mode=retract requires --partial-rollout.")
    if getattr(args, "mask_offpolicy_in_partial_rollout", False) and not getattr(args, "partial_rollout", False):
        raise ValueError("--mask-offpolicy-in-partial-rollout requires --partial-rollout.")


def _validate_dis_args(args) -> None:
    """Validate the paper's direct rollout-policy DIS contract."""
    if getattr(args, "policy_loss_mode", "ppo") != "dis":
        return
    if not args.use_rollout_logprobs:
        raise ValueError("DIS requires --use-rollout-logprobs as its direct behavior-policy anchor.")
    if args.use_tis:
        raise ValueError("DIS is incompatible with --use-tis: its ratio already uses the rollout policy directly.")
    if getattr(args, "eps_clip_c", None) is not None:
        raise ValueError("DIS does not use --eps-clip-c; configure only --eps-clip and --eps-clip-high.")
    if not math.isfinite(args.eps_clip) or not (0.0 < args.eps_clip < 1.0):
        raise ValueError("DIS requires 0 < --eps-clip < 1 so the lower ratio bound is positive.")
    if not math.isfinite(args.eps_clip_high) or args.eps_clip_high <= 0:
        raise ValueError("DIS requires --eps-clip-high > 0.")


def _parse_sequence_mis_args(args) -> None:
    sequence_mis_config = getattr(args, "sequence_mis_config", None)
    if sequence_mis_config is None:
        return

    try:
        config = json.loads(sequence_mis_config)
    except json.JSONDecodeError as exc:
        raise ValueError(
            '--sequence-mis-config must be a JSON object, for example \'{"lower":0.999,"upper":1.001}\'.'
        ) from exc

    if not isinstance(config, dict):
        raise ValueError("--sequence-mis-config must parse to a dictionary/object.")

    allowed_keys = {"aggregation", "lower", "upper", "delta", "token_veto_threshold", "use_advantage", "ratio_source"}
    unknown_keys = set(config) - allowed_keys
    if unknown_keys:
        raise ValueError(f"Unknown --sequence-mis-config keys: {sorted(unknown_keys)}")

    if "aggregation" in config:
        args.sequence_mis_aggregation = str(config["aggregation"])
    if "delta" in config:
        if "lower" in config:
            logger.warning("--sequence-mis-config delta is ignored because lower is also set.")
        else:
            args.sequence_mis_lower = float(config["delta"])
            logger.warning("--sequence-mis-config delta is deprecated; using it as lower.")
    if "lower" in config:
        args.sequence_mis_lower = float(config["lower"])
    if "upper" in config:
        args.sequence_mis_upper = float(config["upper"])
    if "token_veto_threshold" in config:
        args.sequence_mis_token_veto_threshold = float(config["token_veto_threshold"])
    if "use_advantage" in config:
        if not isinstance(config["use_advantage"], bool):
            raise ValueError("--sequence-mis-config use_advantage must be a JSON boolean.")
        args.sequence_mis_use_advantage = config["use_advantage"]
    if "ratio_source" in config:
        ratio_source = str(config["ratio_source"])
        if ratio_source not in {"rollout", "old_actor"}:
            raise ValueError(
                "--sequence-mis-config ratio_source must be 'rollout' (default: sglang-vs-megatron "
                "cross-engine pair) or 'old_actor' (same-stack current-vs-old megatron drift ratio; "
                f"requires --keep-old-actor and is incompatible with routing replay), got {ratio_source!r}."
            )
        args.sequence_mis_ratio_source = ratio_source

    aggregation = getattr(args, "sequence_mis_aggregation", "geometric")
    if aggregation not in {"kl", "geometric", "mirrorpop", "turns_geometric", "turns_mirrorpop"}:
        raise ValueError(
            "--sequence-mis-config aggregation must be one of ['kl', 'geometric', 'mirrorpop', 'turns_geometric', 'turns_mirrorpop'], "
            f"got {aggregation!r}."
        )
    if aggregation in {"turns_geometric", "turns_mirrorpop"} and args.max_turns is None:
        raise ValueError(
            "--max-turns must be set when --sequence-mis-config aggregation=turns_geometric or turns_mirrorpop."
        )
    token_veto_threshold = getattr(args, "sequence_mis_token_veto_threshold", None)
    if token_veto_threshold is not None and token_veto_threshold <= 0:
        raise ValueError(
            "--sequence-mis-config token_veto_threshold must be positive, " f"got {token_veto_threshold}."
        )

    logger.info(
        "Sequence MIS config resolved: aggregation=%s, lower=%s, upper=%s, token_veto_threshold=%s, "
        "use_advantage=%s, config=%s",
        aggregation,
        getattr(args, "sequence_mis_lower", None),
        getattr(args, "sequence_mis_upper", None),
        token_veto_threshold,
        getattr(args, "sequence_mis_use_advantage", False),
        sequence_mis_config,
    )


def _validate_sequence_mis_ratio_source(args) -> None:
    """Same-stack MIS (``ratio_source='old_actor'``) scores drift as current-vs-old
    megatron log-probs, which needs the behavioral old actor AND a current-actor
    recompute at postprocess time. That extra current recompute cannot share routing
    replay's forward-consumption accounting, so it is (for now) mutually exclusive
    with routing replay."""
    if getattr(args, "sequence_mis_ratio_source", "rollout") != "old_actor":
        return
    if not getattr(args, "keep_old_actor", False):
        raise ValueError(
            "--sequence-mis-config ratio_source='old_actor' requires --keep-old-actor "
            "(the behavioral old-actor log-probs are the ratio denominator)."
        )
    if getattr(args, "use_routing_replay", False) or getattr(args, "use_rollout_routing_replay", False):
        raise ValueError(
            "--sequence-mis-config ratio_source='old_actor' is incompatible with routing replay "
            "(--use-routing-replay / --use-rollout-routing-replay): the extra current-actor "
            "recompute would double-consume the replayed routing. Use ratio_source='rollout' "
            "(default) with routing replay, or disable routing replay for same-stack MIS."
        )


def _validate_debug_freeze_old_actor_snapshot(args) -> None:
    """Fail closed around the fixed-behavior debug replay escape hatch.

    Freezing the adapter-only old-actor snapshot is only meaningful when one
    immutable rollout dump is replayed without SGLang or weight publication.
    Keeping the flag valid in any broader mode would silently turn a live PPO
    run's behavioral policy into a stale policy.
    """
    if not getattr(args, "debug_freeze_old_actor_snapshot", False):
        return

    missing = []
    if not getattr(args, "debug_train_only", False):
        missing.append("--debug-train-only")
    if getattr(args, "load_debug_rollout_data", None) is None:
        missing.append("--load-debug-rollout-data")
    if not getattr(args, "keep_old_actor", False):
        missing.append("--keep-old-actor")
    if missing:
        raise ValueError(
            "--debug-freeze-old-actor-snapshot is a fixed-behavior replay-only flag and requires "
            + ", ".join(missing)
            + ". It is forbidden in live rollout/training modes because a frozen old actor would "
            "no longer match the batch's behavior policy."
        )


def _validate_debug_force_old_actor_logprob_recompute(args) -> None:
    """Restrict the matched-forward diagnostic escape hatch to frozen replay."""

    if not getattr(args, "debug_force_old_actor_logprob_recompute", False):
        return

    missing = []
    if not getattr(args, "debug_train_only", False):
        missing.append("--debug-train-only")
    if getattr(args, "load_debug_rollout_data", None) is None:
        missing.append("--load-debug-rollout-data")
    if not getattr(args, "keep_old_actor", False):
        missing.append("--keep-old-actor")
    if missing:
        raise ValueError(
            "--debug-force-old-actor-logprob-recompute is a fixed-replay diagnostic flag and requires "
            + ", ".join(missing)
            + "."
        )


def reset_arg(parser, name, **kwargs):
    """
    Reset the default value of a Megatron argument.
    :param parser: The argument parser.
    :param name: The name of the argument to reset.
    :param default: The new default value.
    """
    for action in parser._actions:
        if name in action.option_strings:
            if "default" in kwargs:
                action.default = kwargs["default"]
            break
    else:
        parser.add_argument(name, **kwargs)


def add_qwen_gdn_arguments(parser):
    """Register Qwen GDN options shared by training and conversion tools."""
    parser.add_argument(
        "--qwen-gdn-backend",
        type=str,
        choices=["fla", "flashqla"],
        default="fla",
        help="GDN implementation backend for Qwen linear-attention layers.",
    )
    parser.add_argument(
        "--qwen-gdn-implementation",
        type=str,
        choices=["replicated", "distributed"],
        default="replicated",
        help=(
            "Qwen GDN rank layout. 'replicated' keeps the HuggingFace-compatible "
            "all-gather path; 'distributed' uses TP-sharded projections and CP "
            "sequence-to-head all-to-all while retaining --qwen-gdn-backend."
        ),
    )
    parser.add_argument(
        "--qwen-gdn-sp-disable-batch-p2p-comm",
        action="store_true",
        help=(
            "Use Megatron's rank-ordered pipeline isend/irecv path for distributed " "Qwen GDN with sequence parallel."
        ),
    )
    return parser


def get_slime_extra_args_provider(add_custom_arguments=None):
    def add_slime_arguments(parser):
        # Ray
        def add_cluster_arguments(parser):
            parser.add_argument("--actor-num-nodes", type=int, default=1, help="Number of nodes for training actor")
            parser.add_argument(
                "--actor-num-gpus-per-node", type=int, default=8, help="Number of gpus per node for training actor"
            )

            parser.add_argument(
                "--rollout-num-gpus",
                type=int,
                default=None,
                help=(
                    "Number of GPUs for inference. Note that when using --colocate, "
                    "i.e. the training and the inference engines are on the same gpus, this param will be ignored and will be set as "
                    "actor_num_gpus_per_node * actor_num_nodes."
                ),
            )
            parser.add_argument(
                "--rollout-num-gpus-per-engine",
                type=int,
                default=1,
                help="Number of GPUs per inference engine, just like the tp_size in sglang.",
            )
            parser.add_argument(
                "--num-gpus-per-node",
                type=int,
                default=8,
                help=(
                    "Number of gpus per node for rollout."
                    "Notice: If you are going to use less than 8 gpus per node under colocate mode, you should set this number."
                ),
            )
            parser.add_argument(
                "--actor-placement-resource",
                type=str,
                default=None,
                help=(
                    "Optional Ray custom resource name required by each actor placement-group GPU bundle. "
                    "Use this to pin training actors to nodes started with matching --resources."
                ),
            )
            parser.add_argument(
                "--rollout-placement-resource",
                type=str,
                default=None,
                help=(
                    "Optional Ray custom resource name required by each rollout placement-group GPU bundle. "
                    "Use this to reserve separate rollout nodes in non-colocated runs."
                ),
            )
            parser.add_argument(
                "--colocate",
                action="store_true",
                default=False,
                help=(
                    "Whether to colocate the inference engines and the actor. "
                    "Turning this on will also set --offload to true."
                ),
            )
            parser.add_argument(
                "--offload",
                action="store_true",
                default=False,
                help=("Equivalent to --offload-train + --offload-rollout. "),
            )
            parser.add_argument(
                "--offload-train",
                action=argparse.BooleanOptionalAction,
                help=(
                    "Whether to offload the training actor to CPU during training. "
                    "This will always be true when --colocate is set."
                ),
            )
            parser.add_argument(
                "--offload-rollout",
                action=argparse.BooleanOptionalAction,
                help=(
                    "Whether to offload the rollout generator to CPU during training. "
                    "This will always be true when --colocate is set."
                ),
            )

            reset_arg(parser, "--distributed-backend", type=str, default="nccl")
            reset_arg(parser, "--distributed-timeout-minutes", type=int, default=10)

            return parser

        def add_train_arguments(parser):
            # --train-backend is parsed early in _pre_parse_mode() and merged later.
            parser.add_argument(
                "--qkv-format",
                type=str,
                choices=["thd", "bshd"],
                default="thd",
                help="The qkv layout for Megatron backend.",
            )
            parser.add_argument(
                "--cp-partition-mode",
                type=str,
                choices=["zigzag", "contiguous"],
                default="zigzag",
                help=(
                    "Context-parallel sequence partition layout. 'zigzag' (default) is Megatron's "
                    "balanced mirror-pair partition. 'contiguous' gives each CP rank a single "
                    "contiguous [r*l_local, (r+1)*l_local) block, required by the DeepSeek-V4-Flash "
                    "attention kernel (raw sliding-window / compressor look-back need a monotone "
                    "local position axis). No-op at --context-parallel-size 1."
                ),
            )
            add_qwen_gdn_arguments(parser)
            parser.add_argument(
                "--fp32-lm-head",
                action="store_true",
                default=False,
                help=(
                    "Run the final LM-head (output_layer) matmul in float32: upcast the hidden states and "
                    "the output_layer weight to fp32 for the vocab projection only. The transformer/MTP "
                    "blocks stay in bf16. Mirrors SGLang's --enable-fp32-lm-head so train- and rollout-time "
                    "log-probs are computed at the same precision, reducing train/infer logits mismatch."
                ),
            )
            parser.add_argument(
                "--train-env-vars",
                type=json.loads,
                default="{}",
                help="Extra environment variables for training process, e.g. PyTorch memory management ones.",
            )
            parser.add_argument(
                "--train-memory-margin-bytes",
                type=int,
                default=1024**3,
                help="Add margin for train memory allocation. By default we will reserve 1GB as margin.",
            )
            parser.add_argument(
                "--megatron-to-hf-mode",
                choices=["raw", "bridge"],
                default="raw",
                help="The method to convert megatron weights to hugging face weights for SGLang.",
            )
            # Delta weight sync.
            parser.add_argument(
                "--update-weight-mode",
                choices=["full", "delta"],
                default="full",
                help=(
                    "Weight sync strategy. 'full' (default) broadcasts every parameter "
                    "every sync. 'delta' detects byte-level changes against a pinned-CPU "
                    "snapshot of the previous broadcast and ships only the changed positions + values."
                ),
            )
            parser.add_argument(
                "--update-weight-transport",
                choices=["nccl", "disk"],
                default="nccl",
                help=(
                    "Per-flush carrier for --update-weight-mode=delta. 'nccl' broadcasts each "
                    "bucket; 'disk' writes each bucket as a safetensors file under "
                    "--update-weight-delta-dir and pushes once at end-of-sync."
                ),
            )
            parser.add_argument(
                "--update-weight-encoding",
                choices=["indices", "deltas", "deltas_zstd"],
                default="indices",
                help=(
                    "Position encoding for partial flushes. 'indices': int32 absolute "
                    "positions (largest, lowest compute). 'deltas': uint16 gap-deltas "
                    "with uint32 fallback (smaller). 'deltas_zstd': 'deltas' with the "
                    "safetensors blob wrapped in zstd L1 (smallest, heaviest compute — "
                    "best for shared-FS bandwidth ≤ ~300 MB/s)."
                ),
            )
            parser.add_argument(
                "--update-weight-delta-dir",
                type=str,
                default=None,
                help=(
                    "Filesystem directory for per-sync delta safetensors. Writable by the "
                    "trainer, readable by every rollout engine. Required when "
                    "--update-weight-transport=disk. One subdirectory per sync "
                    "(``weight_v{N:06d}``), removed after every engine has acknowledged."
                ),
            )
            parser.add_argument(
                "--update-weight-delta-keep-files",
                action="store_true",
                default=False,
                help="Skip post-apply cleanup of per-sync version directories. Useful for debugging.",
            )
            parser.add_argument(
                "--custom-delta-pre-push-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom function called by --update-weight-transport=disk after each "
                    "trainer rank's files are durably on local disk, before rank 0 fires the engine "
                    "RPCs. Signature: ``def hook(args, version_dir: str, rollout_engines) -> None``. "
                    "Called from every trainer rank; the hook gates itself."
                ),
            )
            parser.add_argument(
                "--use-lora-weight-sync",
                action="store_true",
                default=False,
                help=(
                    "LoRA-adapter weight sync (DeepSeek-V4). When set, each sync ships only the "
                    "trainable (requires_grad) LoRA adapter tensors to sglang's "
                    "load_lora_adapter_from_tensors path (base + adapter served) instead of merging "
                    "the adapter into the full base linear and broadcasting it. Default OFF keeps the "
                    "merge-and-broadcast behavior byte-identical. "
                    "Requires the sglang engine launched with --enable-lora and no EAGLE/MTP spec-decode "
                    "(see handoffs/deepseek-v4/lora_serve_design.md)."
                ),
            )
            parser.add_argument(
                "--lora-dim",
                type=int,
                default=0,
                help=(
                    "LoRA adapter rank. A positive value enables LoRA training and "
                    "adapter-only checkpointing; zero disables LoRA."
                ),
            )
            parser.add_argument(
                "--lora-alpha",
                type=int,
                default=None,
                help="LoRA alpha. Defaults to twice --lora-dim when LoRA is enabled.",
            )
            parser.add_argument(
                "--lora-dropout",
                type=float,
                default=0.0,
                help="LoRA adapter dropout probability.",
            )
            parser.add_argument(
                "--lora-rslora",
                action=argparse.BooleanOptionalAction,
                default=False,
                help="Use rank-stabilized LoRA scaling alpha/sqrt(rank).",
            )
            parser.add_argument(
                "--lora-plus-lambda",
                type=float,
                default=None,
                help="LoRA+ B/A learning-rate ratio; unset or 1 disables the split.",
            )
            parser.add_argument(
                "--dsv4-lora-shared-expert",
                action=argparse.BooleanOptionalAction,
                default=False,
                help="Apply LoRA to DS-V4 shared-expert MLPs in addition to attention/compressor linears.",
            )
            parser.add_argument(
                "--lora-adapter-resume-load",
                type=str,
                default="",
                help="Adapter-only checkpoint directory overlaid after loading the frozen base checkpoint.",
            )
            parser.add_argument(
                "--lora-checkpoint-max-node-bytes",
                type=int,
                default=2 * 1024**3,
                help="Maximum node-local size accepted for one adapter-only checkpoint iteration.",
            )
            parser.add_argument(
                "--rollout-lora-name",
                type=str,
                default=None,
                help=(
                    "Name of one statically preloaded SGLang LoRA adapter to attach to every "
                    "rollout /generate request. This is intended for eval/inference with "
                    "--sglang-lora-paths NAME=PATH; dynamic training-time adapter sync should "
                    "leave it unset and use the engine-reported active adapter name instead."
                ),
            )
            parser.add_argument(
                "--custom-model-provider-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom model provider function. "
                    "If set, we will use this function instead of the default model provider. "
                    "The function should have the signature "
                    "`def custom_model_provider(pre_process: bool, post_process: bool, vp_stage: int | None = None) -> GPTModel`. "
                    "Example: 'my_module.my_model_provider'."
                ),
            )
            parser.add_argument(
                "--recompute-loss-function",
                action="store_true",
                help="Whether to disable recompute loss function to save memory during training.",
            )
            parser.add_argument(
                "--log-probs-chunk-size", type=int, default=-1, help="Chunk size to compute log probs to save memory"
            )
            parser.add_argument(
                "--enable-fp32-lm-head",
                action="store_true",
                default=False,
                help=(
                    "Request fp32 lm-head logits. Megatron actor logits are cast to fp32 while preserving the "
                    "original output-layer parameter and TP gradient path; SGLang rollout engines are also asked "
                    "to enable fp32 lm head when supported."
                ),
            )
            parser.add_argument(
                "--only-train-params-name-list",
                type=str,
                nargs="*",
                default=None,
                help="""List of regex patterns of parameter names to TRAIN. All other parameters will be FROZEN. 
                        Supports Python regex syntax (re.search).

                        Examples:
                        1. Train ONLY MoE experts:
                            --only-train-params-name-list experts

                        2. Train ONLY Indexer parameters:
                            --only-train-params-name-list self_attention.wq_b self_attention.wk self_attention.k_norm self_attention.weights_proj

                        3. Train ONLY Layer 20 to 23:
                            --only-train-params-name-list layers\.2[0-3]\.
                        """,
            )

            parser.add_argument(
                "--freeze-params-name-list",
                type=str,
                nargs="*",
                default=None,
                help="""List of regex patterns of parameter names to FREEZE. Other parameters will remain trainable.
                        Supports Python regex syntax (re.search).

                        Examples:
                        1. Freeze Embeddings and Output Layer (common for fine-tuning):
                            --freeze-params-name-list embedding output_layer

                        2. Freeze Indexer parameters:
                            --freeze-params-name-list self_attention.wq_b self_attention.wk self_attention.k_norm self_attention.weights_proj

                        3. Freeze specific projection layers (e.g., all Gate/Up projections):
                            --freeze-params-name-list linear_fc1
                        """,
            )
            parser.add_argument(
                "--allgather-cp",
                action="store_true",
                default=False,
            )

            return parser

        # rollout
        def add_rollout_arguments(parser):
            parser.add_argument(
                "--hf-checkpoint",
                type=str,
                default=None,
                help=(
                    "The huggingface checkpoint of the trained model. "
                    "This is used to initialize sglang and also provide the tokenizer. "
                    "Note that, we will always update the parameters in sglang with that of megatron before training, "
                    "so you only need to provide a huggingface checkpoint that has the same architecture as the model you want to train. "
                    "It doesn't necessary need to contain the most up-to-date parameters."
                ),
            )
            parser.add_argument(
                "--rollout-model-path",
                type=str,
                default=None,
                help=(
                    "Model path served by the rollout engine when it must differ from --hf-checkpoint. "
                    "Used for DeepSeek-V4 DSpark speculative decoding: rollout serves the -DSpark checkpoint "
                    "variant (extra mtp.* draft stages; backbone numerically identical to the standard ckpt, "
                    "see handoffs/deepseek-v4/dspark_backbone_audit.md) while the trainer and weight-sync "
                    "keep using --hf-checkpoint. Defaults to --hf-checkpoint."
                ),
            )
            parser.add_argument(
                "--model-name",
                type=str,
                default=None,
                help=(
                    "The name of the model, this is used to convert the megatron weights into huggingface format. "
                    "If not set, we will use `type(AutoConfig.from_pretrained(args.hf_checkpoint)).__name__.lower()` as model_name. "
                    "Also, sometimes this will help alleviate the bug that transformers cannot find certain model."
                ),
            )
            parser.add_argument(
                "--rollout-function-path",
                type=str,
                default="slime.rollout.sglang_rollout.generate_rollout",
                help=(
                    "Path to the rollout generation function."
                    "You should use this model to create your own custom rollout function, "
                    "and then set this to the path of your custom rollout function. "
                    "The signature of the function should be "
                    "`def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput`"
                    "and within the output sample, you should at least set `tokens`, `response_length`, `reward` "
                    "and `status`."
                ),
            )
            parser.add_argument(
                "--rollout-temperature",
                type=float,
                default=1.0,
                help="the temperature for the inference engine during rollout.",
            )
            parser.add_argument(
                "--rollout-top-p", type=float, default=1.0, help="the top-p for the inference engine during rollout."
            )
            parser.add_argument(
                "--rollout-top-k", type=int, default=-1, help="the top-k for the inference engine during rollout."
            )
            parser.add_argument(
                "--rollout-max-context-len",
                type=int,
                default=None,
                help=(
                    "The maximum context size for the inference engine during rollout."
                    "It should no exceed the `max_position_embeddinds` in Huggingface model's `config.json`"
                ),
            )
            parser.add_argument(
                "--rollout-max-prompt-len",
                type=int,
                default=None,
                help=(
                    "The maximum length of the prompt for the inference engine during rollout. "
                    "If set, we will filter out the long prompts during initialization of the global dataset. "
                    "This is not recommended if the dataset is large."
                ),
            )
            parser.add_argument(
                "--rollout-max-response-len",
                type=int,
                default=None,
                help=(
                    "The maximum length of the response for the inference engine during rollout. "
                    "It is basically `max_tokens` in sglang."
                ),
            )
            parser.add_argument(
                "--rollout-skip-special-tokens",
                action="store_true",
                default=False,
                help=(
                    "Whether to skip special tokens in the response during rollout. "
                    "This is useful when you want to use the response as a prompt for the next rollout."
                ),
            )
            parser.add_argument(
                "--rollout-stop",
                type=str,
                nargs="+",
                default=None,
                help=(
                    "The stop words for the inference engine during rollout. "
                    "It can be a list of strings or a single string. "
                    "It may be hard to pass special tokens in command line, in that case rollout_stop_token_ids can be used."
                ),
            )
            parser.add_argument(
                "--rollout-stop-token-ids",
                type=int,
                nargs="+",
                default=None,
                help=(
                    "The stop token ids for the inference engine during rollout. "
                    "It can be a list of integers or a single integer."
                ),
            )
            parser.add_argument(
                "--rollout-shuffle",
                action="store_true",
                default=False,
                help=("Whether to shuffle the prompts during rollout."),
            )
            parser.add_argument(
                "--rollout-seed",
                type=int,
                default=42,
                help=(
                    "The seed for the random number generator during rollout. "
                    "This is used to shuffle the prompts and also for the random sampling of the prompts."
                ),
            )

            # sampling
            parser.add_argument(
                "--over-sampling-batch-size",
                type=int,
                default=None,
                help=(
                    "This defines the granularity of the sampling batch in the rollout function. "
                    "When the number of available samples falls below the target, a sampling "
                    "operation of size over_sampling_batch_size will be triggered."
                    "Regardless of whether partial rollout is used or filters are applied, "
                    "the sampling granularity is always determined by this value. "
                    "If this value is None, rollout_batch_size will be used as the default over_sampling_batch_size."
                ),
            )
            parser.add_argument(
                "--over-sampling-refill-factor",
                type=int,
                default=None,
                help=(
                    "Optional adaptive refill factor in prompt-group units. When set to F and k accepted "
                    "prompt groups are still missing, submit min(over_sampling_batch_size, F * k) prompt "
                    "groups instead of another fixed over_sampling_batch_size wave. Each prompt group still "
                    "expands to n_samples_per_prompt completions. Default None preserves legacy behavior."
                ),
            )
            parser.add_argument(
                "--dynamic-sampling-filter-path",
                type=str,
                default=None,
                help=(
                    "This is the filter function for dynamic sampling. "
                    "It should be able to judge whether the result of a prompt should be selected or not."
                    "We will do dynamic filter for sampling as in DAPO. e.g. not all correct or all wrong samples."
                    "You could use `slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std` as an example."
                ),
            )
            parser.add_argument(
                "--use-multi-turn",
                action="store_true",
                default=False,
                help=(
                    "Whether to use all turns from a custom multi-turn generate function as training samples. "
                    "When disabled, the custom generate function should return only the last turn sample."
                ),
            )
            parser.add_argument(
                "--filter-by-last-turn",
                action="store_true",
                default=False,
                help=(
                    "When --use-multi-turn is enabled, apply dynamic sampling filter only to the last turn group. "
                    "If the last turn group is kept, all turn groups from the same rollout are kept; otherwise all "
                    "turn groups from that rollout are dropped."
                ),
            )
            parser.add_argument(
                "--max-turns",
                type=int,
                default=None,
                help=(
                    "Maximum number of turns for multi-turn rollout. Also used to scale rollout target group "
                    "counts when --use-multi-turn is enabled."
                ),
            )
            parser.add_argument(
                "--padding-turns",
                action="store_true",
                default=False,
                help=(
                    "Whether custom multi-turn rollout functions should pad missing turns with fake samples. "
                    "The padded samples should be masked out from loss."
                ),
            )
            parser.add_argument(
                "--preserve-history-thinking",
                action="store_true",
                default=False,
                help=(
                    "Let compatible custom multi-turn generate functions build later prompts from the exact "
                    "prompt and generated token IDs of the previous turn. This avoids decode/re-encode drift "
                    "and preserves both historical thinking and rollout prefix-cache locality."
                ),
            )
            # partial rollout
            parser.add_argument(
                "--partial-rollout",
                action="store_true",
                default=False,
                help=(
                    "Whether to use partial rollout. "
                    "If set, the unfinished samples during dynamic sampling will be recycled back to data buffer. "
                    "This is useful for long responses."
                ),
            )
            parser.add_argument(
                "--rollout-weight-sync-pause-mode",
                choices=("abort", "retract"),
                default="abort",
                help=(
                    "How SGLang pauses in-flight generation around a rollout-weight update. "
                    "'abort' returns interrupted requests to the caller; 'retract' parks them, "
                    "releases KV/recurrent state, and re-prefills their retained token prefix "
                    "after generation resumes. Retract requires --partial-rollout."
                ),
            )
            parser.add_argument(
                "--mask-offpolicy-in-partial-rollout",
                action="store_true",
                default=False,
                help=(
                    "Whether to mask previous generation in partial rollout. "
                    "If set, only on-policy generated tokens will be used in training"
                ),
            )
            parser.add_argument(
                "--custom-generate-function-path",
                type=str,
                default=None,
                help=(
                    "Only substitue the `def generate(args, sample, sampling_params)` function within the example rollout function. "
                    "This should be useful if you need to implement some special rollout logic, e.g. multi-turn, function calling."
                ),
            )
            parser.add_argument(
                "--custom-rollout-log-function-path",
                type=str,
                default=None,
                help=(
                    "The custom function for logging rollout data. The signature of the functions is: "
                    "def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool. "
                    "The return value indicates whether to skip the default logging. "
                ),
            )
            parser.add_argument(
                "--log-response-diversity",
                action="store_true",
                default=False,
                help="Whether to log response diversity as unique 4-grams / total 4-grams during rollout.",
            )
            parser.add_argument(
                "--custom-eval-rollout-log-function-path",
                type=str,
                default=None,
                help=(
                    "The custom function for logging eval rollout data. "
                    "def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool. "
                    "The return value indicates whether to skip the default logging. "
                ),
            )

            parser.add_argument(
                "--buffer-filter-path",
                type=str,
                default=None,
                help=(
                    "Path to the buffer filter function. "
                    "It should be able to select the samples in the buffer. "
                    "The function should take list[list[Sample]] and return list[list[Sample]]."
                ),
            )
            # update weight
            parser.add_argument(
                "--update-weight-buffer-size",
                type=int,
                default=512 * 1024**2,
                help=(
                    "buffer size for update weight, in bytes. "
                    "This is used for updating weights by chunk and should be useful for MoE models."
                ),
            )
            parser.add_argument(
                "--update-weights-interval",
                type=int,
                default=1,
                help="Interval for updating the weights",
            )
            parser.add_argument(
                "--keep-old-actor",
                action="store_true",
                help="Whether to keep the rollout model on training process",
            )

            parser.add_argument(
                "--rollout-data-postprocess-path",
                type=str,
                default=None,
                help=(
                    "The called after we have all the rollout data including log_probs. "
                    "It may be helpful for updating loss mask."
                ),
            )
            parser.add_argument(
                "--rollout-external",
                action="store_true",
                default=False,
                help="Use external SGLang instances instead of launching them inside the framework.",
            )
            parser.add_argument(
                "--rollout-external-engine-addrs",
                type=str,
                default=None,
                nargs="+",
                help="Address and ports of the external engines.",
            )
            return parser

        def add_fault_tolerance_arguments(parser):
            parser.add_argument(
                "--use-fault-tolerance",
                action="store_true",
                default=False,
                help="Whether to enable the fault tolerance function during rollout.",
            )
            parser.add_argument(
                "--rollout-health-check-interval",
                type=float,
                default=30.0,
                help="Interval in seconds between rollout engine /health_generate checks during generate/eval.",
            )
            parser.add_argument(
                "--rollout-health-check-timeout",
                type=float,
                default=30.0,
                help="Timeout in seconds to wait for a rollout engine /health_generate response before killing it.",
            )
            parser.add_argument(
                "--rollout-health-check-first-wait",
                type=float,
                default=0,
                help="Initial grace period (in seconds) before starting health checks. This allows time for model compilation and initialization. Increase this value significantly when using deepgemm.",
            )
            return parser

        # data
        def add_data_arguments(parser):
            # dataset
            # TODO: maybe add an num_epoch and calculate the num_rollout from buffer
            parser.add_argument(
                "--num-rollout",
                type=int,
                default=None,
                help="Number of rollout steps. If not set, we will calculate the number of rollout steps from the dataset size.",
            )
            parser.add_argument(
                "--num-epoch",
                type=int,
                default=None,
                help=(
                    "Number of epochs for the training. "
                    "This is used to calculate the number of rollout steps from the dataset size. "
                    "If set, we will calculate the number of rollout steps as `num_rollout = num_epoch * dataset_size // rollout_batch_size`."
                    "If both `--num-epoch` and `--num-rollout` are set, `--num-epoch` will be ignored."
                ),
            )

            parser.add_argument(
                "--disable-rollout-global-dataset",
                action="store_false",
                dest="rollout_global_dataset",
                help=(
                    "Whether to use a global dataset for rollout. "
                    "If set, the rollout will use the `--prompt-data` as the prompt dataset, "
                    "and the prompts for rollout will be sampled from the dataset. "
                    "If not set, you need to manage the data by your self."
                ),
            )

            parser.add_argument(
                "--data-source-path",
                type=str,
                default="slime.rollout.data_source.RolloutDataSourceWithBuffer",
                help="The data source class for rollout data.",
            )
            parser.add_argument(
                "--rollout-dataset-load",
                type=str,
                default=None,
                help=(
                    "Directory from which to load rollout/global_dataset_state_dict_<rollout_id>.pt. "
                    "Defaults to --load for backward compatibility."
                ),
            )
            parser.add_argument(
                "--prompt-data",
                type=str,
                default=None,
                help=(
                    "The path to the prompt data. "
                    "Currently we only support jsonl format, and each line should contains --input-key and --label-key, "
                    "which will be used as the prompt and the label respectively. "
                    "If you want to use a custom template, you can set --apply-chat-template to true, in that case, "
                    "the input should be the same structure as an openai message, e.g. [{'role': 'user', 'content': 'blabla'}]. "
                ),
            )
            parser.add_argument("--apply-chat-template", action="store_true", default=False)
            # Temporarily be JSON-serialized str, will be a real dict after using Omegaconf
            parser.add_argument("--apply-chat-template-kwargs", type=json.loads, default="{}")
            # Extra kwargs forwarded to AutoTokenizer.from_pretrained when loading the rollout
            # tokenizer, e.g. '{"fix_mistral_regex": true}' for Mistral-lineage tokenizers
            # (stepfun Step-3.x) that otherwise drop whitespace under transformers v5.
            parser.add_argument("--tokenizer-load-kwargs", type=json.loads, default="{}")
            parser.add_argument("--input-key", type=str, default="input", help="JSON dataset key")
            parser.add_argument("--label-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument(
                "--multimodal-keys",
                type=json.loads,
                default=None,
                help=(
                    'JSON string for multimodal data mapping media types to data keys. Example: \'{"image": "image_file"}\''
                ),
            )
            parser.add_argument("--metadata-key", type=str, default="metadata", help="JSON dataset key")
            parser.add_argument(
                "--tool-key",
                type=str,
                default="tools",
                help=(
                    "When need to add tools during apply_chat_template, you should provide the key for the tools in the prompt dataset."
                ),
            )

            parser.add_argument(
                "--start-rollout-id",
                type=int,
                default=None,
                help=(
                    "The starting rollout step, if not set, will try to load the step from --load when doing continue training, "
                    "otherwise will be set to 0, meaning training from start."
                ),
            )

            # batch sizes
            parser.add_argument(
                "--rollout-batch-size",
                type=int,
                required=True,
                help=(
                    "The number of prompts in each rollout step. "
                    "The total data returned should be rollout_batch_size * n_samples_per_prompt. "
                ),
            )
            parser.add_argument(
                "--n-samples-per-prompt", type=int, default=1, help="Number of responses for each prompt in generation"
            )

            # gbs of the training, note that the gbs is of sample, not of prompts,
            # so if you hope to train 1 step for each rollout, the global_bach_size should be set as
            # `rollout_batch_size * n_samples_per_prompt`.
            reset_arg(parser, "--global-batch-size", type=int, default=None)
            parser.add_argument(
                "--num-steps-per-rollout",
                type=int,
                default=None,
                help=(
                    "Number of steps per rollout, e.g. It is equivalent to setting gbs as "
                    "`rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout`."
                ),
            )
            # mbs for the training, will be ignored if `use_dynamic_batch_size` is set.
            reset_arg(parser, "--micro-batch-size", type=int, default=1)
            parser.add_argument(
                "--balance-data",
                action="store_true",
                default=False,
                help=(
                    "Balance the number of tokens between data parallel ranks with `karmarkar_karp` for verl. "
                    "Note that this may allocate the different response of the same prompt into different training steps."
                ),
            )
            parser.add_argument(
                "--enable-turns-dp-partitions",
                action="store_true",
                default=False,
                help=(
                    "Split DP data by complete multi-turn trajectories. Samples are ordered by sample_indices "
                    "and turn_indices before partitioning; with --balance-data, balancing is done at trajectory granularity."
                ),
            )
            parser.add_argument(
                "--sort-train-microbatches-by-padded-length-desc",
                action="store_true",
                default=False,
                help=(
                    "Within each training step and DP rank, execute already-assigned microbatches from "
                    "largest to smallest padded sequence width. This does not change DP balancing or "
                    "the theoretical peak of the longest microbatch; it can reduce allocator growth and "
                    "fragmentation by establishing the largest allocation first. With turn-aware DP "
                    "partitions, this currently requires one training sample per trajectory."
                ),
            )

            parser.add_argument(
                "--use-dynamic-batch-size",
                action="store_true",
                default=False,
                help=(
                    "Because the sample length varies, to maximize the GPU utilization, "
                    "we will use the dynamic batch size to adjust the micro batch size according to the maximum number of tokens each gpu can run. "
                    "For example, if we have 3 samples, with the length of 100, 200, and 300, and the max_tokens_per_gpu is 300, when enabling "
                    "dynamic batch size, slime will make 2 micro batches, i.e. [100, 200], [300]."
                ),
            )
            parser.add_argument(
                "--max-tokens-per-gpu",
                type=int,
                default=None,
                help=(
                    "The maximum number of tokens per GPU for dynamic batch size. "
                    "Note that when enabling context parallel (CP), the max tokens per gpu should be around "
                    "`max_response_len // cp_size` instead of `max_response_len`."
                ),
            )
            parser.add_argument(
                "--log-probs-max-tokens-per-gpu",
                type=int,
                default=None,
                help=(
                    "The maximum number of tokens per GPU for calculating log probs. "
                    "This is used to calculate the log probs of the responses during rollout, "
                    "and should be set to a larger value than `max_tokens_per_gpu` if you want better performance. "
                ),
            )
            return parser

        def add_eval_arguments(parser):
            parser.add_argument(
                "--eval-function-path",
                type=str,
                default=None,
                help=(
                    "Path to the eval generation function."
                    "If not set, we will use rollout_function_path as the default. "
                ),
            )

            # change the default value of eval_interval from Megatron to None
            reset_arg(parser, "--eval-interval", type=int, default=None)

            parser.add_argument(
                "--eval-prompt-data",
                type=str,
                default=None,
                nargs="+",
                help=(
                    "Path to the evaluation prompt data, "
                    "should first input the name of the eval dataset and then the path, e.g. "
                    "aime /path/to/aime.jsonl"
                ),
            )
            parser.add_argument(
                "--eval-config",
                type=str,
                default=None,
                help=(
                    "Path to an OmegaConf YAML/JSON file describing evaluation datasets. "
                    "When provided, this overrides --eval-prompt-data."
                ),
            )
            parser.add_argument(
                "--skip-eval-before-train",
                action="store_true",
                default=False,
                help="Whether to skip evaluation before training.",
            )

            # The following keys are used to override the rollout version during eval.
            parser.add_argument("--eval-input-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument("--eval-label-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument("--eval-tool-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument(
                "--n-samples-per-eval-prompt",
                type=int,
                default=1,
                help="number of responses for each prompt in generation",
            )
            parser.add_argument("--eval-temperature", type=float, default=None)
            parser.add_argument("--eval-top-p", type=float, default=None)
            parser.add_argument("--eval-top-k", type=int, default=None)
            parser.add_argument("--eval-max-response-len", type=int, default=None)
            parser.add_argument("--eval-max-prompt-len", type=int, default=None)
            parser.add_argument("--eval-min-new-tokens", type=int, default=None)
            parser.add_argument("--eval-max-context-len", type=int, default=None)

            return parser

        def add_algo_arguments(parser):
            parser.add_argument(
                "--ref-load",
                type=str,
                default=None,
                help=(
                    "The checkpoint for reference model. "
                    "When --load is not set, this will be used as the initial checkpoint for training. "
                ),
            )
            parser.add_argument(
                "--ref-ckpt-step", type=int, default=None, help="The checkpoint step for reference model. "
            )
            reset_arg(parser, "--load", type=str, default=None)
            reset_arg(parser, "--save", type=str, default=None)
            reset_arg(parser, "--save-interval", type=int, default=None)
            reset_arg(parser, "--async-save", action="store_true")
            reset_arg(
                parser,
                "--no-save-optim",
                action="store_true",
                default=False,
                help=(
                    "If set, do not save the optimizer state when saving checkpoints. "
                    "This reduces checkpoint size but disables training resumption from the saved checkpoint."
                ),
            )
            parser.add_argument(
                "--save-hf",
                type=str,
                default=None,
                help=(
                    "Path to save the model in HuggingFace format when using Megatron backend. "
                    "The model will be saved to `save_hf.format(rollout_id)`. "
                    "In raw Megatron-to-HF mode, weights are saved with the same quantization config "
                    "as `--hf-checkpoint`. "
                ),
            )
            reset_arg(parser, "--seed", type=int, default=1234)
            reset_arg(parser, "--clip-grad", type=float, default=1.0)
            reset_arg(parser, "--calculate-per-token-loss", action="store_true")
            reset_arg(parser, "--lr", type=float, default=1e-6)

            parser.add_argument(
                "--num-critic-only-steps",
                type=int,
                default=0,
                help="Number of initial rollout steps that train critic only; set >= num_rollout for critic-only runs",
            )
            parser.add_argument(
                "--megatron-config-path",
                type=str,
                default=None,
                help=(
                    "Path to a structured YAML config for Megatron roles. The file should use "
                    "a top-level 'megatron' key with role-tagged entries; the critic runtime will "
                    "select exactly one entry with role=critic. Legacy 'critic' configs are still accepted."
                ),
            )

            parser.add_argument("--eps-clip", type=float, default=0.2, help="PPO clip range")
            parser.add_argument(
                "--policy-loss-mode",
                type=str,
                default="ppo",
                choices=[
                    "ppo",
                    "dis",
                    "up",
                    "aspo",
                    "ripo",
                    "dppo_binary_tv",
                    "dppo_binary_kl",
                    "dppo_topk_kl_predictive",
                    "cppo",
                    "cispo",
                    "drpo",
                ],
                help=(
                    "Policy-loss trust region: PPO/DIS, UP/ASPO/RIPO, binary or predictive DPPO, "
                    "CPPO, CISPO, and DRPO. DPPO modes use eps-clip/eps-clip-high as divergence "
                    "thresholds; dppo_topk_kl_predictive uses rollout Top-K support."
                ),
            )
            parser.add_argument(
                "--dis-ratio-level",
                choices=["token", "sequence"],
                default="token",
                help="Compute DIS importance ratios per token or from each response's mean log-ratio.",
            )
            parser.add_argument(
                "--dppo-predictive-top-k",
                type=int,
                default=0,
                help=(
                    "Number of behavior-policy Top-K entries stored per response token for "
                    "dppo_topk_kl_predictive. The sampled token is added to the support when "
                    "it is not already in Top-K, so storage width is K+1."
                ),
            )
            parser.add_argument(
                "--dppo-predictive-tail-estimator",
                type=str,
                choices=["aggregated", "uniform"],
                default="aggregated",
                help="Tail approximation for the predictive Top-K KL directional derivative.",
            )
            parser.add_argument(
                "--ripo-delta",
                type=float,
                default=0.05,
                help="RIPO/RIC trust-region radius delta. Paper arXiv:2607.10169 uses 0.05 by default.",
            )
            parser.add_argument(
                "--ripo-delta-high",
                type=float,
                default=None,
                help="Optional RIPO upper-bound delta, analogous to --eps-clip-high. Defaults to --ripo-delta.",
            )
            parser.add_argument(
                "--ripo-ratio-min",
                type=float,
                default=0.5,
                help="Outer lower bound for RIPO importance-ratio clipping. Paper uses 0.5.",
            )
            parser.add_argument(
                "--ripo-ratio-max",
                type=float,
                default=10.0,
                help="Outer upper bound for RIPO importance-ratio clipping. Paper uses 10.",
            )
            parser.add_argument(
                "--cppo-prefix-delta",
                type=float,
                default=0.02,
                help=(
                    "CPPO dynamic prefix-budget floor delta_b_min. Each sequence uses "
                    "clamp(P90(D), delta_b_min, 2*delta_b_min), matching the official UniRL implementation. "
                    "The paper uses 0.015 for the post-trained model and 0.02 for Base models."
                ),
            )
            parser.add_argument(
                "--cppo-weight-floor",
                type=float,
                default=0.8,
                help=(
                    "CPPO final-token position weight w_min; weights decay linearly from 1. "
                    "CPPO (arXiv:2606.10968) uses 0.8."
                ),
            )
            parser.add_argument(
                "--eps-clip-high",
                type=float,
                default=None,
                help="PPO clip upper offset; the final ratio upper bound is 1 + eps_clip_high.",
            )
            parser.add_argument(
                "--eps-clip-c",
                type=float,
                default=None,
                help=(
                    "Dual-clip threshold. For PPO it is the lower bound from https://arxiv.org/pdf/1912.09729; "
                    "for ASPO it is the optional upper bound for soft dual-clipping positive reciprocal weights; "
                    "for DPPO/CPPO it is the detached importance-ratio upper bound."
                ),
            )
            parser.add_argument("--value-clip", type=float, default=0.2, help="the clip for value loss")
            parser.add_argument(
                "--kl-coef",
                type=float,
                default=0.00,
                help="KL penalty coefficient for reward shaping. This is applied to the reward signal before advantage calculation.",
            )
            parser.add_argument(
                "--loss-type",
                type=str,
                choices=["policy_loss", "sft_loss", "custom_loss"],
                default="policy_loss",
                help=(
                    "Choose loss type, currently support ppo policy_loss or sft_loss, "
                    "if custom_loss is set, we will use the function path from `--custom-loss-function-path`."
                ),
            )
            parser.add_argument(
                "--custom-loss-function-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom loss function, if the loss_type is `custom_loss`, "
                    "we will use this function to calculate the loss. "
                ),
            )
            parser.add_argument(
                "--kl-loss-type",
                type=str,
                choices=["k1", "k2", "k3", "low_var_kl"],
                default="k1",
                help="Choose KL loss type: kl, k2, k3, low_var_kl",
            )
            parser.add_argument(
                "--advantage-estimator",
                type=str,
                choices=[
                    "grpo",
                    "gspo",
                    "reinforce_plus_plus",
                    "reinforce_plus_plus_baseline",
                    "ppo",
                    "rloo",
                    "trloo",
                ],
                default="grpo",
                help=(
                    "Advantage estimator to use. Note: on-policy distillation (OPD) is now orthogonal "
                    "to the advantage estimator. Use --opd-kl-coef > 0 to enable OPD on top of any estimator."
                ),
            )
            parser.add_argument(
                "--multi-turn-gamma",
                type=float,
                default=1.0,
                help="Discount factor for multi-turn reward folding used by multi-turn advantage estimators.",
            )
            parser.add_argument(
                "--disable-compute-advantages-and-returns",
                action="store_false",
                dest="compute_advantages_and_returns",
                help=(
                    "Whether to disable computing advantages and returns. "
                    "If set, we will not compute the advantages and returns, "
                    "This is useful for sft or custom loss function."
                ),
            )
            parser.add_argument(
                "--custom-advantage-function-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom advantage/returns computation function. "
                    "When set, this function replaces the built-in compute_advantages_and_returns. "
                    "Signature: def custom_fn(args, rollout_data) -> None. "
                    "The function should set rollout_data['advantages'] and rollout_data['returns'] in-place. "
                    "Critic values are available in rollout_data['values']. "
                    "(e.g., my_module.py:my_advantage_fn)."
                ),
            )
            parser.add_argument(
                "--use-kl-loss", action="store_true", default=False, help="whether to use KL loss from GRPO"
            )
            parser.add_argument(
                "--kl-loss-coef",
                type=float,
                default=0.0,
                help="KL penalty coefficient for the loss function. This is added to the final PPO loss.",
            )
            parser.add_argument(
                "--use-unbiased-kl",
                action="store_true",
                default=False,
                help="Whether to enable unbiased KL estimation.",
            )
            parser.add_argument(
                "--ref-update-interval",
                type=int,
                default=None,
                help="Interval (in rollout steps) to update ref model from actor. If None, ref model is not updated.",
            )
            parser.add_argument("--entropy-coef", type=float, default=0.0, help="Entropy loss coef")
            parser.add_argument("--gamma", type=float, default=1.0, help="PPO GAE gamma")
            parser.add_argument("--lambd", type=float, default=1.0, help="PPO GAE lambd")
            parser.add_argument("--normalize-advantages", action="store_true", default=False)
            parser.add_argument(
                "--disable-grpo-std-normalization",
                action="store_false",
                dest="grpo_std_normalization",
                help="from Dr.GRPO https://arxiv.org/pdf/2503.20783",
            )
            parser.add_argument(
                "--disable-rewards-normalization",
                action="store_false",
                dest="rewards_normalization",
                help="Disable rewards normalization",
            )
            parser.add_argument(
                "--use-rollout-entropy",
                action="store_true",
                default=False,
                help=(
                    "Whether to calculate the entropy when calculating the logprobs from actor and reference model. "
                    "This is useful for doing special loss mask."
                ),
            )
            parser.add_argument(
                "--entropy-common-probe",
                action="store_true",
                default=False,
                help="Measure entropy on the original fixed-batch response mask for diagnostic comparisons.",
            )
            parser.add_argument(
                "--assert-zero-lora-out",
                action="store_true",
                default=False,
                help="Require a freshly loaded LoRA actor to have an exact-zero adapter output.",
            )
            parser.add_argument(
                "--get-mismatch-metrics",
                action="store_true",
                default=False,
                help="Whether to calculate the mismatch metrics.",
            )
            parser.add_argument(
                "--reset-optimizer-states",
                action="store_true",
                default=False,
                help=(
                    "Whether to reset optimizer states after each rollout. "
                    "If enabled, the optimizer's history will be cleared at the end of each rollout, which can sometimes help with training stability or fulfill specific experiment requirements."
                ),
            )
            parser.add_argument(
                "--use-rollout-logprobs",
                action="store_true",
                default=False,
                help=(
                    "Whether to use the rollout logprobs when calculating the importance sampling ratios. "
                    "If not set, we will use the logprobs from the actor model."
                ),
            )
            # Off-Policy Correction using Importance Sampling: https://fengyao.notion.site/off-policy-rl
            parser.add_argument(
                "--use-tis",
                action="store_true",
                default=False,
                help="Enable TIS from https://fengyao.notion.site/off-policy-rl for off-policy importance sampling.",
            )
            parser.add_argument(
                "--tis-clip",
                type=float,
                default=2.0,
                help="Clipping threshold C for importance sampling ratios to control variance.",
            )
            parser.add_argument(
                "--tis-clip-low",
                type=float,
                default=0,
                help="Lower bound clipping threshold C for importance sampling ratios to control variance.",
            )
            parser.add_argument(
                "--custom-tis-function-path",
                type=str,
                default=None,
                help="Path to the custom TIS/RS function (e.g., examples/train_infer_mismatch_helper/mis.py:compute_mis_weights_with_cp).",
            )
            parser.add_argument(
                "--custom-pg-loss-reducer-function-path",
                type=str,
                default=None,
                help="Path to a custom reducer function for pg_loss only. When set, pg_loss will use this custom reducer while other metrics (pg_clipfrac, ppo_kl, entropy_loss, etc.) still use the default sum_of_sample_mean. (e.g., examples/Dr.GRPO/custom_reducer.py:get_pg_loss_reducer).",
            )

            parser.add_argument(
                "--use-routing-replay",
                action="store_true",
                default=False,
                help="The routing replay technique from https://arxiv.org/abs/2507.18071",
            )
            parser.add_argument(
                "--use-rollout-routing-replay",
                action="store_true",
                default=False,
                help="The rollout routing replay technique from https://arxiv.org/abs/2510.11370",
            )
            parser.add_argument(
                "--use-opsm",
                action="store_true",
                default=False,
                help="Whether to enable Off-Policy Sequence Masking (OPSM).",
            )
            parser.add_argument(
                "--opsm-delta",
                type=float,
                default=1e-4,
                help="Sequence-level KL threshold for Off-Policy Sequence Masking (OPSM).",
            )
            parser.add_argument(
                "--sequence-mis-config",
                type=str,
                default=None,
                help=(
                    "Optional Sequence MIS config for rollout-data postprocess masking. "
                    "Supports aggregation, lower/upper thresholds, token veto, and use_advantage keys. "
                    'Must be a JSON object, for example \'{"aggregation":"turns_geometric","lower":0.999,"upper":1.001}\'.'
                ),
            )
            return parser

        def add_on_policy_distillation_arguments(parser):
            """Add on-policy distillation (OPD) related arguments.

            OPD is orthogonal to advantage estimators and can be applied on top of
            any estimator (GRPO, PPO, etc.) by adding a KL penalty to advantages.
            """
            parser.add_argument(
                "--use-opd",
                action="store_true",
                default=False,
                help="Enable on-policy distillation (OPD). Must specify --opd-type when enabled.",
            )
            parser.add_argument(
                "--opd-type",
                type=str,
                choices=["sglang", "megatron"],
                default=None,
                help=(
                    "Type of on-policy distillation. "
                    "'sglang': Teacher log-probs are obtained from external SGLang server during rollout. "
                    "'megatron': Teacher model is loaded via --opd-teacher-load and forwarded during training."
                ),
            )
            parser.add_argument(
                "--opd-kl-coef",
                type=float,
                default=1.0,
                help="On-policy distillation KL penalty coefficient. Default is 1.0.",
            )
            parser.add_argument(
                "--opd-teacher-load",
                type=str,
                default=None,
                help=(
                    "The checkpoint for OPD teacher model. Required when --opd-type=megatron. "
                    "The teacher model should have the same architecture as policy/ref model."
                ),
            )
            parser.add_argument(
                "--opd-teacher-ckpt-step", type=int, default=None, help="The checkpoint step for OPD teacher model."
            )
            return parser

        def add_router_arguments(parser):
            parser.add_argument(
                "--use-slime-router",
                action="store_true",
                default=False,
                help="Whether to use SlimeRouter for text-based routing instead of SGLang token-based routing",
            )
            RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
            return parser

        # wandb
        def add_wandb_arguments(parser):
            # wandb parameters
            parser.add_argument("--use-wandb", action="store_true", default=False)
            parser.add_argument(
                "--wandb-mode",
                type=str,
                default=None,
                choices=["online", "offline", "disabled"],
                help="W&B mode: online (default), offline (local only), or disabled. Overrides WANDB_MODE env var.",
            )
            parser.add_argument(
                "--wandb-dir",
                type=str,
                default=None,
                help="Directory to store wandb logs. Default is ./wandb in current directory.",
            )
            parser.add_argument("--wandb-key", type=str, default=None)
            parser.add_argument("--wandb-host", type=str, default=None)
            parser.add_argument("--wandb-team", type=str, default=None)
            parser.add_argument("--wandb-group", type=str, default=None)
            reset_arg(parser, "--wandb-project", type=str, default=None)
            parser.add_argument(
                "--disable-wandb-random-suffix",
                action="store_false",
                dest="wandb_random_suffix",
                default=True,
                help=(
                    "Whether to add a random suffix to the wandb run name. "
                    "By default, we will add a random 6 length string with characters to the run name."
                ),
            )
            parser.add_argument(
                "--wandb-always-use-train-step",
                action="store_true",
                default=False,
                help=(
                    "Whether to always use train step as the step metric in wandb. "
                    "If set, we will always use the train steps for wandb logging, "
                    "otherwise, will use rollout step for most info other than train/*. "
                ),
            )
            parser.add_argument(
                "--wandb-centralized",
                action="store_true",
                default=False,
                help=(
                    "Route W&B and TensorBoard logging through a single Ray actor on the driver node. "
                    "This avoids multiple Ray actors writing to the same W&B run."
                ),
            )
            parser.add_argument(
                "--log-multi-turn",
                action="store_true",
                default=False,
                help="Whether to log information for multi-turn rollout.",
            )
            parser.add_argument(
                "--log-passrate",
                action="store_true",
                default=False,
                help="Whether to turn on passrate logging, which will log the pass@n of the responses in the rollout.",
            )
            parser.add_argument(
                "--log-reward-category",
                type=str,
                default=None,
                help=(
                    "Log statistics of the category of reward, such as why the reward function considers it as failed. "
                    "Specify the key in the reward dict using this argument."
                ),
            )
            parser.add_argument(
                "--log-correct-samples",
                action="store_true",
                default=False,
                help="Whether to turn on passrate logging, which will log the pass@n of the responses in the rollout.",
            )
            parser.add_argument("--wandb-run-id", type=str, default=None)
            return parser

        # tensorboard
        def add_tensorboard_arguments(parser):
            # tb_project_name, tb_experiment_name
            parser.add_argument("--use-tensorboard", action="store_true", default=False)
            parser.add_argument(
                "--tb-project-name",
                type=str,
                default=None,
                help="Directory to store tensorboard logs. Default is  os.environ.get('TENSORBOARD_DIR') directory.",
            )
            parser.add_argument("--tb-experiment-name", type=str, default=None)

            return parser

        # debug
        def add_debug_arguments(parser):
            parser.add_argument(
                "--save-debug-rollout-data",
                type=str,
                default=None,
                help=(
                    "Save the rollout data to this path for debugging. "
                    "The file will be saved to `save_debug_rollout_data.format(rollout_id)`."
                ),
            )
            # --load-debug-rollout-data, --debug-rollout-only, --debug-train-only
            # are parsed early in _pre_parse_mode() and merged later.
            parser.add_argument(
                "--load-forge-rollout-data",
                type=str,
                default=None,
                help=(
                    "Path (or {rollout_id} template) to a dumped rollout .pt file replayed by "
                    "slime.rollout.forge_load.generate_rollout. Mirrors --load-debug-rollout-data's "
                    "format(rollout_id=...) convention: a path without the placeholder is treated as "
                    "a literal file and reused across every rollout_id; a path containing {rollout_id} "
                    "loads a per-rollout file (with eval_<id>.pt for the eval pipeline). Unlike "
                    "--load-debug-rollout-data, this does NOT force debug_train_only / skip_sglang -- "
                    "sglang servers, router, weight_update and the colocate offload/onload dance all "
                    "stay live, which is the point (memory measurement at long context)."
                ),
            )
            parser.add_argument(
                "--load-debug-rollout-data-subsample",
                type=float,
                default=None,
                help="Subsample a portion of the debug rollout data for faster debugging.",
            )
            parser.add_argument(
                "--debug-freeze-old-actor-snapshot",
                action="store_true",
                default=False,
                help=(
                    "Fixed-behavior debug replay only: seed the LoRA old-actor adapter snapshot once "
                    "from the initial live actor and never refresh it across repeated training passes. "
                    "Requires --debug-train-only, --load-debug-rollout-data, and --keep-old-actor."
                ),
            )
            parser.add_argument(
                "--debug-force-old-actor-logprob-recompute",
                action="store_true",
                default=False,
                help=(
                    "Fixed-replay denominator A/B only: execute the frozen old-actor log-prob forward "
                    "even when --use-rollout-logprobs selects the rollout denominator. This equalizes "
                    "the forward path without enabling TIS or mismatch metrics. Requires "
                    "--debug-train-only, --load-debug-rollout-data, and --keep-old-actor."
                ),
            )
            parser.add_argument(
                "--save-debug-train-data",
                type=str,
                default=None,
                help=(
                    "Save the train data to this path for debugging. "
                    "The file will be saved to `save_debug_train_data.format(rollout_id)`."
                ),
            )
            parser.add_argument(
                "--dump-details",
                type=str,
                default=None,
                help=("Dump all details of training for post-hoc analysis and visualization."),
            )
            # use together with --record-memory-history and --memory-snapshot-path (defined in Megatron)
            parser.add_argument(
                "--memory-snapshot-dir",
                type=str,
                default=".",
            )
            parser.add_argument(
                "--memory-snapshot-num-steps",
                type=int,
                default=None,
            )
            parser.add_argument(
                "--profile-target",
                type=str,
                choices=["train_overall", "train_actor", "train_log_probs"],
                default=["train_overall"],
                nargs="+",
            )
            parser.add_argument(
                "--memory-recorder",
                type=str,
                choices=["torch", "memray"],
                default="torch",
            )
            reset_arg(parser, "--record-memory-history", action="store_true", default=False)
            parser.add_argument("--check-weight-update-equal", action="store_true")
            return parser

        def add_network_arguments(parser):
            parser.add_argument("--http-proxy", type=str, default=None)
            parser.add_argument("--use-distributed-post", action="store_true", default=False)
            return parser

        def add_reward_model_arguments(parser):
            parser.add_argument(
                "--rm-type",
                type=str,
                default=None,
                help="Type of the reward model",
            )
            parser.add_argument(
                "--reward-key",
                type=str,
                default=None,
                help=(
                    "Some reward model may return a dict instead of a value, "
                    "this is the key to extract the reward value from the dict. "
                ),
            )
            parser.add_argument(
                "--eval-reward-key",
                type=str,
                default=None,
                help="The eval variant for --reward-key",
            )
            parser.add_argument(
                "--group-rm", action="store_true", default=False, help="Whether to do rm on a whole group."
            )
            parser.add_argument(
                "--rm-url",
                type=str,
                default=None,
                help="URL for the reward model service for --rm-type remote_rm, e.g. http://localhost:8000",
            )
            parser.add_argument(
                "--custom-rm-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom reward model function. "
                    "If set, we will use this function to calculate the reward instead of the default one. "
                    "The function should have the signature `def custom_rm(args, sample) -> float`."
                ),
            )
            parser.add_argument(
                "--custom-reward-post-process-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom function that will post process reward, by default it will be the normalization for grpo. "
                ),
            )
            parser.add_argument(
                "--custom-convert-samples-to-train-data-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom function that converts samples to training data. "
                    "If set, this function will replace the default _convert_samples_to_train_data. "
                    "The function should have the signature `def convert_samples_to_train_data(args, samples) -> dict`."
                ),
            )
            return parser

        def add_kernel_agent_arguments(parser):
            parser.add_argument(
                "--kernel-env-url",
                dest="kernel_env_url",
                type=str,
                default=None,
                help="Kernel agent environment server URL, e.g. http://127.0.0.1:8002.",
            )
            parser.add_argument(
                "--kernel-backend",
                type=str,
                choices=["cuda_agent", "tvm_ffi", "triton"],
                default="cuda_agent",
                help="Kernel backend name sent to the kernel agent environment.",
            )
            parser.add_argument(
                "--reference-backend",
                type=str,
                choices=["torch", "torch_compile"],
                default="torch",
                help="Reference backend name sent to the kernel agent environment.",
            )
            parser.add_argument(
                "--do-precheck",
                action=argparse.BooleanOptionalAction,
                default=True,
                help="Whether the kernel agent should run client-side precheck before env execution.",
            )
            parser.add_argument(
                "--use-reference-cache",
                action="store_true",
                default=False,
                help=(
                    "Ask KernelGym to reuse cached reference timing (KernelGym /evaluate "
                    "use_reference_cache). The cache key is derived from the reference identity "
                    "(entry_point + ground_truth hash), so a reference is timed once instead of "
                    "on every kernel attempt. Only safe for fixed-input references."
                ),
            )
            parser.add_argument(
                "--finalize-mode",
                type=str,
                choices=["none", "positive", "improve"],
                default="positive",
                help=(
                    "Kernel agent turn finalization mode. 'none' keeps all generated turn samples; "
                    "'positive' removes non-positive turns after a positive best reward; "
                    "'improve' removes turns that do not improve the best reward."
                ),
            )
            parser.add_argument(
                "--use-coverage-rs",
                action="store_true",
                default=False,
                help="Whether to enable coverage-based rejection sampling for kernel-agent turn samples.",
            )
            parser.add_argument(
                "--first-turn-max-context-len",
                type=int,
                default=None,
                help=(
                    "Optional prompt-plus-response context cap for turn 0 of kernel-agent multi-turn rollout. "
                    "Later turns continue to use --rollout-max-context-len."
                ),
            )
            parser.add_argument(
                "--coverage-rs-key",
                type=str,
                choices=["time_coverage", "num_coverage"],
                default="time_coverage",
                help="Coverage metric used by kernel-agent coverage-based rejection sampling.",
            )
            parser.add_argument(
                "--coverage-rs-threshold",
                type=float,
                default=0.3,
                help="Coverage threshold used by kernel-agent coverage-based rejection sampling.",
            )
            parser.add_argument(
                "--coverage-rs-factor",
                type=float,
                default=0.1,
                help="Linear keep-probability factor used by kernel-agent coverage-based rejection sampling.",
            )
            parser.add_argument(
                "--overlong-penalty",
                action="store_true",
                default=False,
                help="Apply a linear reward penalty near the response-length cap.",
            )
            parser.add_argument(
                "--overlong-buffer-len",
                type=int,
                default=2048,
                help="Number of tokens before the response cap over which the overlong penalty ramps.",
            )
            parser.add_argument(
                "--overlong-penalty-factor",
                type=float,
                default=1.0,
                help="Maximum reward subtraction applied by --overlong-penalty.",
            )
            parser.add_argument(
                "--overlong-penalty-turn-idx",
                type=int,
                default=None,
                help=(
                    "Optional zero-based turn index to which --overlong-penalty is restricted. "
                    "The default applies the penalty to every turn."
                ),
            )
            parser.add_argument(
                "--overlong-use-effective-response-cap",
                action="store_true",
                default=False,
                help=(
                    "Compute the overlong window from min(response cap, context cap - prompt length). "
                    "Opt in when the serving response limit is clamped by the remaining context."
                ),
            )
            parser.add_argument(
                "--use-conditional-truncation-mask",
                action="store_true",
                default=False,
                help=(
                    "Enable Conditional Truncation Masking from MicroCoder-GRPO (arXiv:2603.07777), which "
                    "probabilistically zeros post-processed advantages for eligible max-length responses."
                ),
            )
            parser.add_argument(
                "--conditional-truncation-mask-prob",
                type=float,
                default=0.1,
                help=("CTM masking probability rho. The paper compares 0.1, 0.2, and 0.3; slime defaults to 0.1."),
            )
            return parser

        def add_rollout_buffer_arguments(parser):
            parser.add_argument(
                "--rollout-buffer-url",
                type=str,
                default=None,
                help="URL for the rollout buffer",
            )

            parser.add_argument(
                "--fetch-trajectory-retry-times",
                type=int,
                default=-1,
                help="Number of times to retry fetching trajectory, -1 means unlimited retry",
            )
            parser.add_argument(
                "--min-batch-collection-ratio",
                type=float,
                default=1,
                help="Minimum batch collection ratio",
            )
            parser.add_argument(
                "--rollout-task-type",
                type=str,
                default="math",
            )
            parser.add_argument(
                "--loss-mask-type",
                type=str,
                default="qwen",
                choices=["qwen", "qwen3", "qwen3_5", "distill_qwen"],
                help="Loss mask type",
            )
            parser.add_argument(
                "--data-pad-size-multiplier",
                type=int,
                default=128,
                help="Multiplier for data padding size in data processing.",
            )
            parser.add_argument(
                "--rollout-sample-filter-path",
                type=str,
                default=None,
                help=(
                    "Path to the rollout sample filter function. "
                    "This function determines whether a sample will participate in loss calculation. "
                    "The function should take args and samples (list[Sample]) as input, and return None. "
                    "Please directly modify the remove_sample attribute of Sample. "
                    "Note: This attribute does not determine whether the sample participates in advantage normalization."
                ),
            )
            parser.add_argument(
                "--rollout-all-samples-process-path",
                type=str,
                default=None,
                help=(
                    "Path to the rollout all samples process function that "
                    "can process all samples including filtered ones."
                ),
            )
            return parser

        def add_custom_megatron_plugins_arguments(parser):
            """
            Add custom Megatron plugins arguments.
            This is a placeholder for any additional arguments that might be needed.
            """
            # Custom arguments can be added here
            parser.add_argument(
                "--custom-megatron-init-path",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--custom-megatron-before-log-prob-hook-path",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--custom-megatron-before-train-step-hook-path",
                type=str,
                default=None,
            )
            return parser

        def add_mtp_training_arguments(parser):
            """Add MTP training specific arguments."""
            reset_arg(parser, "--mtp-num-layers", type=int, default=None)
            reset_arg(parser, "--mtp-loss-scaling-factor", type=float, default=0.2)
            parser.add_argument(
                "--enable-mtp-training",
                action="store_true",
                default=False,
                help="Enable MTP layer parameter updates during training",
            )

            return parser

        def add_ci_arguments(parser):
            parser.add_argument(
                "--ci-test",
                action="store_true",
            )
            parser.add_argument(
                "--ci-disable-kl-checker",
                action="store_true",
            )
            parser.add_argument(
                "--ci-save-grad-norm",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--ci-load-grad-norm",
                type=str,
                default=None,
            )
            return parser

        # Add custom arguments in front to prevent overwritten some slime arguments.
        if add_custom_arguments is not None:
            parser = add_custom_arguments(parser)

        parser = add_cluster_arguments(parser)
        parser = add_train_arguments(parser)
        parser = add_rollout_arguments(parser)
        parser = add_fault_tolerance_arguments(parser)
        parser = add_data_arguments(parser)
        parser = add_eval_arguments(parser)
        parser = add_algo_arguments(parser)
        parser = add_on_policy_distillation_arguments(parser)
        parser = add_wandb_arguments(parser)
        parser = add_tensorboard_arguments(parser)
        parser = add_router_arguments(parser)
        parser = add_debug_arguments(parser)
        parser = add_network_arguments(parser)
        parser = add_reward_model_arguments(parser)
        parser = add_kernel_agent_arguments(parser)
        parser = add_rollout_buffer_arguments(parser)
        parser = add_mtp_training_arguments(parser)
        parser = add_ci_arguments(parser)
        parser = add_custom_megatron_plugins_arguments(parser)
        reset_arg(
            parser,
            "--custom-config-path",
            type=str,
            default=None,
            help="Path to the YAML config for custom function arguments.",
        )
        reset_arg(
            parser,
            "--multi-turn-prompt-config-path",
            type=str,
            default=None,
            help="Path to the YAML config for multi-turn prompt templates.",
        )
        reset_arg(parser, "--padded-vocab-size", type=int, default=None)

        return parser

    return add_slime_arguments


def _pre_parse_mode():
    """Pre-parse CLI to extract arguments that control parsing flow.

    These arguments are removed from add_slime_arguments to avoid
    registering them twice.  The returned namespace is merged into
    the final ``args`` after Phase 2 parsing.
    """
    temp_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    temp_parser.add_argument("--train-backend", type=str, choices=["megatron"], default="megatron")
    temp_parser.add_argument("--debug-rollout-only", action="store_true", default=False)
    temp_parser.add_argument("--debug-train-only", action="store_true", default=False)
    temp_parser.add_argument("--load-debug-rollout-data", type=str, default=None)
    temp_args, _ = temp_parser.parse_known_args()
    return temp_args


def parse_args(add_custom_arguments=None):
    # Users may call `parse_args` very early, thus we ensure logger is configured here
    configure_logger()

    add_slime_arguments = get_slime_extra_args_provider(add_custom_arguments)

    pre = _pre_parse_mode()
    skip_sglang = pre.debug_train_only or pre.load_debug_rollout_data is not None

    # Phase 1: Parse sglang args independently (separate parser, parse_known_args).
    # Skipped when sglang servers are not needed.
    sglang_ns = None
    if not skip_sglang:
        sglang_ns = sglang_parse_args()

    # Phase 2: Parse megatron + slime args.
    # Uses ignore_unknown_args=True so that --sglang-* and pre-parsed CLI flags
    # are silently ignored by the megatron parser.
    from slime.backends.megatron_utils.arguments import megatron_parse_args
    from slime.backends.megatron_utils.arguments import validate_args as megatron_validate_args

    args = megatron_parse_args(
        extra_args_provider=add_slime_arguments,
        skip_hf_validate=pre.debug_rollout_only,
    )

    # Merge pre-parsed args into the main namespace
    for key, value in vars(pre).items():
        setattr(args, key, value)

    # Merge sglang args into the main namespace
    if sglang_ns is not None:
        for key, value in vars(sglang_ns).items():
            setattr(args, key, value)

    slime_validate_args(args)

    if pre.train_backend == "megatron" and not args.debug_rollout_only:
        megatron_validate_args(args)

    if not args.debug_train_only:
        sglang_validate_args(args)

    return args


def _apply_megatron_role_overrides(base_args, overrides, role):
    role_args = copy.deepcopy(base_args)
    ignored_keys = {"num_nodes", "num_gpus_per_node"}

    # Apply overrides from the YAML config.
    # Unspecified keys inherit from base_args via deepcopy.
    for key, value in overrides.items():
        if key in ignored_keys:
            logger.info(f"Ignoring {role} config key '{key}'; GPU allocation always follows CLI args.")
            continue
        if not hasattr(role_args, key):
            logger.warning(f"{role.capitalize()} config key '{key}' is not a known argument, setting it anyway.")
        else:
            # YAML safe_load doesn't parse scientific notation (e.g. 1e-5) as float.
            # Coerce the value to match the type of the existing attribute.
            original = getattr(role_args, key)
            if original is not None and isinstance(value, str) and isinstance(original, (int, float)):
                try:
                    value = type(original)(value)
                except (ValueError, TypeError):
                    pass
        setattr(role_args, key, value)

    if role == "critic":
        # Critic-specific: disable features that only apply to actors.
        role_args.kl_coef = 0
        role_args.use_opd = False
        role_args.custom_advantage_function_path = None
        role_args.untie_embeddings_and_output_weights = True
        if "disable_param_buffers_cpu_backup" not in overrides:
            role_args.disable_param_buffers_cpu_backup = False

    return role_args


def parse_megatron_role_args(base_args, megatron_config_path, role):
    """Parse role-specific arguments from a unified Megatron YAML config.

    The config must contain a top-level ``megatron`` list with per-role entries.
    Missing roles inherit the base args unchanged.
    """
    assert role in {"actor", "critic"}, f"Unsupported Megatron config role: {role}"

    with open(megatron_config_path) as f:
        raw_config = yaml.safe_load(f) or {}

    assert "megatron" in raw_config, (
        "megatron config must contain a top-level 'megatron' list, e.g. "
        "megatron: [{name: default, role: actor, overrides: {...}}]"
    )

    overrides = {}
    megatron_entries = raw_config["megatron"]
    assert isinstance(megatron_entries, list), (
        "megatron config 'megatron' field must be a list, e.g. "
        "megatron: [{name: default, role: actor, overrides: {...}}]"
    )
    role_entries = [entry for entry in megatron_entries if entry.get("role") == role]
    assert len(role_entries) <= 1, (
        f"megatron config must contain at most one entry with role={role}, e.g. "
        f"megatron: [{{name: default, role: {role}, overrides: {{...}}}}]"
    )
    if role_entries:
        role_entry = role_entries[0]
        overrides = role_entry.get("overrides") or role_entry.get("args") or {}
    else:
        logger.info(
            f"No megatron config entry with role={role} found in {megatron_config_path}; using inherited args."
        )

    role_args = _apply_megatron_role_overrides(base_args, overrides, role)
    logger.info(
        f"Parsed megatron config for role={role} from {megatron_config_path}: overrides = {list(overrides.keys())}"
    )

    return role_args


def parse_critic_args(actor_args, megatron_config_path):
    """Backward-compatible wrapper for critic-specific Megatron role parsing."""
    return parse_megatron_role_args(actor_args, megatron_config_path, role="critic")


def _resolve_eval_datasets(args) -> list[EvalDatasetConfig]:
    """
    Build evaluation dataset configurations from either --eval-config or --eval-prompt-data.
    """
    datasets_config = []
    defaults: dict[str, Any] = {}

    if args.eval_config:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(args.eval_config)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg_dict, dict):
            raise ValueError("--eval-config must contain a mapping at the root.")

        eval_cfg = cfg_dict.get("eval", cfg_dict)
        if not isinstance(eval_cfg, dict):
            raise ValueError("--eval-config must define an `eval` mapping or be a mapping itself.")

        defaults = dict(eval_cfg.get("defaults") or {})
        datasets_config = ensure_dataset_list(eval_cfg.get("datasets"))
        if not datasets_config:
            raise ValueError("--eval-config does not define any datasets under `eval.datasets`.")
    elif args.eval_prompt_data:
        values = list(args.eval_prompt_data)
        if len(values) == 1:
            logger.info("[legacy] only one eval_prompt_data detected, will assume it is data for aime")
            values = ["aime", values[0]]
        if len(values) % 2 != 0:
            raise ValueError("eval prompt data must be provided as name/path pairs.")
        datasets_config = [{"name": values[i], "path": values[i + 1]} for i in range(0, len(values), 2)]
    else:
        datasets_config = []

    eval_datasets = build_eval_dataset_configs(args, datasets_config, defaults)
    if eval_datasets:
        args.eval_prompt_data = [item for dataset in eval_datasets for item in (dataset.name, dataset.path)]
    else:
        args.eval_prompt_data = None

    return eval_datasets


def _resolve_checkpoint_load_args(args) -> None:
    """Apply checkpoint fallbacks without replacing an explicit rollout id."""
    if args.megatron_to_hf_mode == "bridge":
        load_is_megatron = (
            args.load is not None
            and os.path.exists(args.load)
            and os.path.exists(os.path.join(args.load, "latest_checkpointed_iteration.txt"))
        )
        if not load_is_megatron:
            if args.load is None:
                args.load = args.ref_load or args.hf_checkpoint
            if args.start_rollout_id is None:
                args.start_rollout_id = 0
        return

    load_is_megatron = (
        args.load is not None
        and os.path.exists(args.load)
        and os.path.exists(os.path.join(args.load, "latest_checkpointed_iteration.txt"))
    )
    if not load_is_megatron:
        args.no_load_optim = True
        args.no_load_rng = True
        args.finetune = True
        args.load = args.ref_load
        if args.ref_ckpt_step is not None:
            args.ckpt_step = args.ref_ckpt_step
        if args.start_rollout_id is None:
            args.start_rollout_id = 0


def slime_validate_args(args):
    if getattr(args, "enable_fp32_lm_head", False):
        args.fp32_lm_head = True
    if getattr(args, "fp32_lm_head", False):
        args.enable_fp32_lm_head = True
        if not getattr(args, "sglang_enable_fp32_lm_head", False):
            logger.info("fp32 LM head is set; enabling fp32 LM head for rollout engines.")
        args.sglang_enable_fp32_lm_head = True

    _parse_sequence_mis_args(args)
    _validate_lora_args(args)
    # rollout_temperature <= 0 (greedy) breaks the train-side log-prob path,
    # which divides logits by the temperature to match rollout log-probs:
    # /0 -> Inf -> NaN loss -> "found NaN in local grad norm" in the first
    # backward. Only rollout-only debugging (no train side) may use it.
    if getattr(args, "rollout_temperature", 1.0) <= 0 and not getattr(args, "debug_rollout_only", False):
        raise ValueError(
            f"--rollout-temperature must be > 0 for training (got {args.rollout_temperature}): "
            "the train-side log-prob computation divides logits by the rollout temperature, "
            "so 0 (greedy) produces a NaN loss. Use --rollout-temperature 1 for on-policy RL; "
            "temperature 0 is only allowed with --debug-rollout-only."
        )
    if getattr(args, "sequence_mis_aggregation", None) in {"turns_geometric", "turns_mirrorpop"} and not getattr(
        args, "enable_turns_dp_partitions", False
    ):
        raise ValueError(
            "--enable-turns-dp-partitions must be set when Sequence MIS aggregation is "
            "turns_geometric or turns_mirrorpop."
        )
    _validate_sequence_mis_ratio_source(args)
    args.eval_datasets = _resolve_eval_datasets(args)

    conditional_truncation_mask_prob = getattr(args, "conditional_truncation_mask_prob", 0.1)
    assert 0.0 <= conditional_truncation_mask_prob <= 1.0, "conditional_truncation_mask_prob must be in [0, 1]."
    if getattr(args, "use_conditional_truncation_mask", False):
        reward_post_process_path = getattr(args, "custom_reward_post_process_path", None)
        expected_path = "examples.kernel_agent.kernel_reward.reward_post_process_by_group"
        if reward_post_process_path != expected_path:
            logger.warning(
                "--use-conditional-truncation-mask is applied by %s, but "
                "--custom-reward-post-process-path is %r. CTM will not be applied unless the configured hook "
                "implements equivalent post-normalization masking.",
                expected_path,
                reward_post_process_path,
            )

    if args.use_slime_router:
        logger.warning(
            "--use-slime-router is deprecated and ignored. slime now always uses sglang_router "
            "built from https://github.com/zhuzilin/sgl-router."
        )
        args.use_slime_router = False

    if args.kl_coef != 0 or args.use_kl_loss:
        if not os.path.exists(args.ref_load):
            raise FileNotFoundError(f"ref_load {args.ref_load} does not exist, please check the path.")

        if not os.path.exists(os.path.join(args.ref_load, "latest_checkpointed_iteration.txt")):
            logger.info(
                f"ref_load {args.ref_load} does not have latest_checkpointed_iteration.txt, "
                "please make sure it is a valid megatron checkpoint directory."
            )

    # Validate on-policy distillation (OPD) arguments
    if args.use_opd:
        if args.opd_type is None:
            raise ValueError("--opd-type must be specified when --use-opd is enabled. Choose 'sglang' or 'megatron'.")

        if args.opd_type == "megatron":
            if args.opd_teacher_load is None:
                raise ValueError(
                    "--opd-teacher-load is required when --opd-type=megatron. "
                    "Please provide the path to the teacher model checkpoint."
                )
            if not os.path.exists(args.opd_teacher_load):
                raise FileNotFoundError(
                    f"opd_teacher_load {args.opd_teacher_load} does not exist, please check the path."
                )
            if not os.path.exists(os.path.join(args.opd_teacher_load, "latest_checkpointed_iteration.txt")):
                logger.info(
                    f"opd_teacher_load {args.opd_teacher_load} does not have latest_checkpointed_iteration.txt, "
                    "please make sure it is a valid megatron checkpoint directory."
                )

        elif args.opd_type == "sglang":
            if args.opd_teacher_load is not None:
                raise ValueError(
                    "--opd-teacher-load should not be set when --opd-type=sglang. "
                    "In sglang mode, teacher log-probs are obtained from external server during rollout."
                )
    else:
        # If OPD is not enabled, opd_teacher_load should not be set
        if args.opd_teacher_load is not None:
            raise ValueError("--opd-teacher-load is set but --use-opd is not enabled. Please add --use-opd flag.")

    _resolve_checkpoint_load_args(args)

    if args.eval_interval is not None:
        assert args.eval_datasets, "Evaluation datasets must be configured when eval_interval is set."

    if args.save_interval is not None:
        assert args.save is not None, "'--save' is required when save_interval is set."

    assert not (args.kl_coef != 0 and args.kl_loss_coef != 0), "Only one of kl_coef and kl_loss_coef can be set"

    if args.advantage_estimator in ["reinforce_plus_plus", "reinforce_plus_plus_baseline"]:
        assert args.normalize_advantages, (
            "The 'reinforce_plus_plus' and 'reinforce_plus_plus_baseline' advantage estimators "
            "require advantage normalization. Please add `--normalize-advantages` to your command."
        )

    if args.use_rollout_logprobs:
        assert not args.use_tis, "use_rollout_logprobs and use_tis cannot be set at the same time."

    if args.use_multi_turn and args.custom_reward_post_process_path is None:
        logger.warning(
            "--use-multi-turn can produce uneven turn groups. Configure --custom-reward-post-process-path "
            "for turn-aware reward normalization, for example "
            "examples.kernel_agent.kernel_reward.reward_post_process_by_group."
        )
    if args.preserve_history_thinking and not args.use_multi_turn:
        raise ValueError("--preserve-history-thinking requires --use-multi-turn.")
    _validate_partial_rollout_args(args)
    if args.overlong_penalty_turn_idx is not None:
        if args.overlong_penalty_turn_idx < 0:
            raise ValueError("--overlong-penalty-turn-idx must be non-negative.")
        if args.max_turns is not None and args.overlong_penalty_turn_idx >= args.max_turns:
            raise ValueError("--overlong-penalty-turn-idx must be smaller than --max-turns.")

    if args.get_mismatch_metrics:
        assert (
            args.custom_tis_function_path is not None
        ), "custom_tis_function_path must be set when get_mismatch_metrics is set"

        if args.use_rollout_logprobs:
            logger.info(
                "get_mismatch_metrics is set; For metrics calculation, the log probs will still be recomputed by training engine. One more forward pass will be applied."
            )

    if args.use_dynamic_batch_size:
        assert args.max_tokens_per_gpu is not None, "max_tokens_per_gpu must be set when use_dynamic_batch_size is set"
        if args.log_probs_max_tokens_per_gpu is None:
            args.log_probs_max_tokens_per_gpu = args.max_tokens_per_gpu

    policy_loss_mode = getattr(args, "policy_loss_mode", "ppo")
    use_dppo_binary = policy_loss_mode in ["dppo_binary_tv", "dppo_binary_kl"]
    if policy_loss_mode == "cppo":
        if not math.isfinite(args.eps_clip) or args.eps_clip <= 0.0:
            raise ValueError(f"--eps-clip must be finite and positive for CPPO, got {args.eps_clip}.")
        if not math.isfinite(args.cppo_prefix_delta) or args.cppo_prefix_delta <= 0.0:
            raise ValueError(f"--cppo-prefix-delta must be finite and positive, got {args.cppo_prefix_delta}.")
        if not math.isfinite(args.cppo_weight_floor) or not 0.0 < args.cppo_weight_floor <= 1.0:
            raise ValueError(f"--cppo-weight-floor must be finite and in (0, 1], got {args.cppo_weight_floor}.")
        if args.eps_clip_high is not None and args.eps_clip_high != args.eps_clip:
            logger.warning(
                "CPPO uses --eps-clip as its symmetric TV-divergence threshold; --eps-clip-high=%s is ignored.",
                args.eps_clip_high,
            )
        if args.eps_clip_c is not None and (not math.isfinite(args.eps_clip_c) or args.eps_clip_c <= 1.0):
            raise ValueError(
                "--eps-clip-c must be finite and greater than 1 for CPPO's detached ratio upper bound, "
                f"got {args.eps_clip_c}."
            )
    if policy_loss_mode == "ripo":
        if not math.isfinite(args.ripo_delta) or args.ripo_delta <= 0.0:
            raise ValueError(f"--ripo-delta must be a finite positive number, got {args.ripo_delta}.")
        if args.ripo_delta_high is not None and (
            not math.isfinite(args.ripo_delta_high) or args.ripo_delta_high <= 0.0
        ):
            raise ValueError(
                f"--ripo-delta-high must be a finite positive number when set, got {args.ripo_delta_high}."
            )
        if not math.isfinite(args.ripo_ratio_min) or not 0.0 <= args.ripo_ratio_min <= 1.0:
            raise ValueError(f"--ripo-ratio-min must be finite and in [0, 1], got {args.ripo_ratio_min}.")
        if not math.isfinite(args.ripo_ratio_max) or args.ripo_ratio_max < 1.0:
            raise ValueError(f"--ripo-ratio-max must be finite and >= 1, got {args.ripo_ratio_max}.")
        if args.ripo_ratio_min > args.ripo_ratio_max:
            raise ValueError(
                "--ripo-ratio-min must not exceed --ripo-ratio-max, got "
                f"{args.ripo_ratio_min} > {args.ripo_ratio_max}."
            )
    assert not (
        (use_dppo_binary or policy_loss_mode in {"cppo", "drpo"}) and args.use_tis
    ), "DPPO binary loss, CPPO, DRPO, and TIS apply policy-loss corrections; disable use_tis."
    if policy_loss_mode in {"dppo_binary_tv", "dppo_binary_kl", "cppo", "drpo"} and not args.use_rollout_logprobs:
        if policy_loss_mode in {"dppo_binary_tv", "dppo_binary_kl"}:
            loss_name = "DPPO binary loss"
        elif policy_loss_mode == "cppo":
            loss_name = "CPPO"
        else:
            loss_name = policy_loss_mode.upper()
        paper_context = (
            " This is the decoupled-objective ablation, not the rollout behavior-policy anchor used in "
            "the DPPO paper (https://arxiv.org/abs/2602.04879)."
            if policy_loss_mode in {"dppo_binary_tv", "dppo_binary_kl"}
            else (
                " CPPO (https://arxiv.org/abs/2606.10968) defines its divergence against the rollout "
                "behavior policy."
                if policy_loss_mode == "cppo"
                else ""
            )
        )
        logger.warning(
            "%s is using actor-recomputed old log_probs.%s Add --use-rollout-logprobs to use the rollout "
            "behavior-policy anchor and skip the actor old-logprob forward pass.",
            loss_name,
            paper_context,
        )

    if args.eps_clip_high is None:
        args.eps_clip_high = args.eps_clip

    _validate_dppo_predictive_args(args)
    _validate_dis_args(args)

    if args.eval_reward_key is None:
        args.eval_reward_key = args.reward_key

    if args.dump_details is not None:
        args.save_debug_rollout_data = f"{args.dump_details}/rollout_data/{{rollout_id}}.pt"
        args.save_debug_train_data = f"{args.dump_details}/train_data/{{rollout_id}}_{{rank}}.pt"

    if args.load_debug_rollout_data is not None:
        logger.info(
            f"load_debug_rollout_data {args.load_debug_rollout_data} is set, "
            "will not instantiate sglang servers and will only run the training process."
        )
        args.debug_train_only = True

    # Validate after --load-debug-rollout-data has implied debug_train_only.
    # Doing this near the top of slime_validate_args would reject the valid
    # implicit form before that normalization has happened.
    _validate_debug_freeze_old_actor_snapshot(args)
    _validate_debug_force_old_actor_logprob_recompute(args)

    args.use_critic = args.advantage_estimator == "ppo"
    # Critic always uses the same GPU count as actor.
    args.critic_num_gpus_per_node = args.actor_num_gpus_per_node
    args.critic_num_nodes = args.actor_num_nodes

    if args.offload:
        args.offload_train = True
        args.offload_rollout = True
    del args.offload

    if args.debug_rollout_only:
        if args.colocate and (not args.rollout_num_gpus):
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
        else:
            args.actor_num_gpus_per_node = min(8, args.rollout_num_gpus)
            args.actor_num_nodes = args.rollout_num_gpus // args.actor_num_gpus_per_node
        args.colocate = False
        args.offload_train = args.offload_rollout = False
        if args.train_memory_margin_bytes > 0:
            logger.warning("Force train_memory_margin_bytes=0 since debug_rollout_only does not support it")
            args.train_memory_margin_bytes = 0

    assert not (args.debug_rollout_only and args.debug_train_only), (
        "debug_rollout_only and debug_train_only cannot be set at the same time, " "please set only one of them."
    )

    # always true on offload for colocate at the moment.
    if args.colocate:
        if args.offload_train is None:
            args.offload_train = True
        if args.offload_rollout is None:
            args.offload_rollout = True
        if args.rollout_num_gpus != args.actor_num_gpus_per_node * args.actor_num_nodes:
            logger.info(
                f"rollout_num_gpus {args.rollout_num_gpus} != actor_num_gpus_per_node {args.actor_num_gpus_per_node} "
                f"* actor_num_nodes {args.actor_num_nodes}, overriding rollout_num_gpus to match actor_num_gpus_per_node * actor_num_nodes."
            )
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes

    if args.offload_train is None:
        args.offload_train = False
    if args.offload_rollout is None:
        args.offload_rollout = False

    if args.use_critic:
        args.offload_train = True

    if args.offload_train:
        args.disable_grad_buffers_cpu_backup = True
        args.disable_param_buffers_cpu_backup = True

    if args.eval_function_path is None:
        args.eval_function_path = args.rollout_function_path

    if args.num_steps_per_rollout is not None:
        global_batch_size = args.rollout_batch_size * args.n_samples_per_prompt // args.num_steps_per_rollout
        if args.global_batch_size is not None:
            assert args.global_batch_size == global_batch_size, (
                f"global_batch_size {args.global_batch_size} is not equal to "
                f"rollout_batch_size {args.rollout_batch_size} * n_samples_per_prompt {args.n_samples_per_prompt} "
                f"// num_steps_per_rollout {args.num_steps_per_rollout}"
            )
        args.global_batch_size = global_batch_size

    if args.n_samples_per_prompt == 1:
        args.grpo_std_normalization = False
        logger.info("n_samples_per_prompt is set to 1, grpo_std_normalization will be set to False.")

    if args.over_sampling_batch_size is None:
        args.over_sampling_batch_size = args.rollout_batch_size

    assert args.over_sampling_batch_size >= args.rollout_batch_size, (
        f"over_sampling_batch_size {args.over_sampling_batch_size} should be greater than or equal to "
        f"rollout_batch_size {args.rollout_batch_size}"
    )
    over_sampling_refill_factor = getattr(args, "over_sampling_refill_factor", None)
    if over_sampling_refill_factor is not None and over_sampling_refill_factor < 1:
        raise ValueError(
            "over_sampling_refill_factor must be a positive integer when configured, "
            f"got {over_sampling_refill_factor}"
        )

    if args.num_epoch is not None:
        if args.num_rollout is not None:
            logger.info("Both num_epoch and num_rollout are set, num_epoch will be ignored.")
        else:
            assert args.rollout_global_dataset, (
                "num_epoch is set, but rollout_global_dataset is not set, "
                "please remove --disable-rollout-global-dataset to use num_epoch"
            )
    else:
        # if num_epoch is not set, we should set num_rollout
        assert args.num_rollout is not None, (
            "num_epoch is not set, but num_rollout is not set, " "please set --num-rollout or --num-epoch"
        )

    if args.enable_mtp_training:
        assert args.mtp_num_layers, "mtp_num_layers must be set when enable_mtp_training is set"

    if args.use_rollout_routing_replay:
        args.use_routing_replay = True

    if args.custom_config_path:
        with open(args.custom_config_path) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if hasattr(args, k):
                logger.info(f"Warning: Argument {k} is already set to {getattr(args, k)}, will override with {v}.")
            setattr(args, k, v)
        _validate_partial_rollout_args(args)

    if args.eval_max_context_len is None:
        logger.info(
            f"args.eval_max_context_len is not set. Use args.rollout_max_context_len {args.rollout_max_context_len} as default value."
        )
        args.eval_max_context_len = args.rollout_max_context_len

    if args.rollout_max_context_len is not None:
        if args.rollout_max_prompt_len is None:
            args.rollout_max_prompt_len = args.rollout_max_context_len - 1
            logger.info(
                f"args.rollout_max_prompt_len is not set. Use args.rollout_max_context_len - 1 ({args.rollout_max_context_len} - 1) as default value so that there is at least one generated token to compute loss."
            )
        assert (
            args.rollout_max_prompt_len <= args.rollout_max_context_len - 1
        ), f"args.rollout_max_prompt_len ({args.rollout_max_prompt_len}) must be smaller than args.rollout_max_context_len ({args.rollout_max_context_len}) so that there is at least one generated token to compute loss."

    if args.first_turn_max_context_len is not None:
        assert args.use_multi_turn, "--first-turn-max-context-len requires --use-multi-turn."
        assert args.first_turn_max_context_len > 0, "--first-turn-max-context-len must be positive."
        assert (
            args.rollout_max_context_len is not None
        ), "--first-turn-max-context-len requires --rollout-max-context-len."
        assert args.first_turn_max_context_len <= args.rollout_max_context_len, (
            f"--first-turn-max-context-len ({args.first_turn_max_context_len}) must not exceed "
            f"--rollout-max-context-len ({args.rollout_max_context_len})."
        )

    if args.qkv_format == "bshd":
        assert args.train_backend == "megatron", "bshd format is only supported for megatron backend."
        assert (
            args.use_dynamic_batch_size is False
        ), "Dynamic batch size is not supported for bshd format. Please specify --micro-batch-size instead."

    if args.only_train_params_name_list and args.freeze_params_name_list:
        raise ValueError("You can only specify ONE of: --only-train-params-name-list, or --freeze-params-name-list.")

    if args.update_weight_mode == "delta":
        if args.colocate:
            raise ValueError(
                "--update-weight-mode=delta is not supported with --colocate. Colocate transfers "
                "weights via CUDA IPC (only a handle crosses processes), so the delta bookkeeping "
                "(snapshot + diff + sparse encode) is pure overhead."
            )
        if args.update_weight_transport == "disk" and not args.update_weight_delta_dir:
            raise ValueError(
                "--update-weight-transport=disk requires --update-weight-delta-dir to point at "
                "a filesystem shared between the trainer and the rollout engines."
            )
