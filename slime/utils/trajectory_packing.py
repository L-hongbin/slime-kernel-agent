"""Pack turn-level TRLOO training records into one causal sequence per trajectory.

Rollout filtering and reward normalization must run before this transformation.
The input history and the targets scored by the loss are separate: a terminal
token moved by a chat-template repair is still scored from its original prefix.
"""

from collections import defaultdict

import numpy as np


def validate_trajectory_packing_args(args) -> None:
    if not getattr(args, "pack_multi_turn_trajectories", False):
        return
    if not args.use_multi_turn or args.advantage_estimator != "trloo":
        raise ValueError("--pack-multi-turn-trajectories requires --use-multi-turn and --advantage-estimator trloo")
    if not args.custom_reward_post_process_path:
        raise ValueError("--pack-multi-turn-trajectories requires turn-aware --custom-reward-post-process-path")
    if getattr(args, "enable_mtp_training", False):
        if getattr(args, "spec", None) != ["slime_plugins.models.qwen3_5", "get_qwen3_5_spec"]:
            raise ValueError("Trajectory packing with MTP training requires the native Qwen3.5/Qwen3.8 spec")
        if not getattr(args, "calculate_per_token_loss", False):
            raise ValueError("Trajectory packing with MTP training requires --calculate-per-token-loss")
    if getattr(args, "loss_type", "policy_loss") != "policy_loss" or getattr(args, "policy_loss_mode", "ppo") not in {
        "ppo",
        "dppo_topk_kl_predictive",
    }:
        raise ValueError("--pack-multi-turn-trajectories supports PPO and predictive Top-K DPPO policy loss")
    # These extensions interpret each training record as a separate response,
    # or require model state beyond the shared causal token history.
    unsupported = (
        "use_tis",
        "get_mismatch_metrics",
        "use_opsm",
        "use_opd",
        "use_critic",
        "use_routing_replay",
        "use_rollout_routing_replay",
        "allgather_cp",
        "lora_dim",
        "log_correct_samples",
        "log_passrate",
        "custom_convert_samples_to_train_data_path",
        "custom_advantage_function_path",
        "rollout_data_postprocess_path",
        "custom_pg_loss_reducer_function_path",
    )
    for name in unsupported:
        if getattr(args, name, None):
            raise ValueError(f"--pack-multi-turn-trajectories does not support --{name.replace('_', '-')}")
    for name in ("attention_dropout", "hidden_dropout"):
        if getattr(args, name, 0.0):
            raise ValueError(f"--pack-multi-turn-trajectories requires --{name.replace('_', '-')} 0 for equivalence")


def pack_multi_turn_trajectories(data: dict, samples: list) -> dict:
    """Coalesce normalized turn samples; reject any changed scored prefix.

    Response arrays span everything after the first prompt, with zero-mask gaps
    for observations/template text. ``token_rewards`` holds the already computed
    per-turn TRLOO advantages, before optional train-side whitening.
    """
    unsupported = {"multimodal_train_inputs", "rollout_routed_experts", "metadata", "teacher_log_probs"} & data.keys()
    if unsupported:
        raise ValueError(f"Trajectory packing cannot merge fields: {sorted(unsupported)}")
    if "turn_indices" not in data:
        raise ValueError("Trajectory packing requires turn_indices")
    trajectories = defaultdict(list)
    for i, sample_index in enumerate(data["sample_indices"]):
        if sample_index is None or data["turn_indices"][i] is None:
            raise ValueError("Trajectory packing requires non-null sample_indices and turn_indices")
        trajectories[sample_index].append(i)

    packed = {key: [] for key in data}
    packed.update(target_tokens=[], token_rewards=[], loss_normalization_counts=[], packed_turn_metrics=[])
    response_fields = (
        "rollout_log_probs",
        "rollout_topk_token_ids",
        "rollout_topk_log_probs",
        "rollout_topk_valid_mask",
    )
    metric_fields = ("rewards", "raw_reward", "truncated", "turn_indices")
    for sample_index, positions in trajectories.items():
        positions.sort(key=lambda i: data["turn_indices"][i])
        turns = [data["turn_indices"][i] for i in positions]
        if len(set(turns)) != len(turns):
            raise ValueError(f"Duplicate turns in trajectory {sample_index}: {turns}")
        if len({data["group_ids"][i] for i in positions}) != 1:
            raise ValueError(f"Inconsistent training group in trajectory {sample_index}")
        real = [i for i in positions if not samples[i].metadata.get("is_pad_turn", False)]
        if not real:
            real = positions[:1]
        last = real[-1]
        tokens = list(data["tokens"][last])
        prompt_length = min(len(data["tokens"][i]) - data["response_lengths"][i] for i in real)
        if prompt_length < 1:
            raise ValueError(f"Trajectory {sample_index} must have at least one prompt token")
        response_length = len(tokens) - prompt_length
        targets = tokens.copy()
        mask = [0] * response_length
        token_rewards = [0.0] * response_length
        fields = {}
        for key in response_fields:
            if key in data:
                example = np.asarray(data[key][last])
                fields[key] = np.zeros((response_length, *example.shape[1:]), dtype=example.dtype)
        for i in positions:
            turn_mask = data["loss_masks"][i]
            if not any(turn_mask):
                continue
            source = data["tokens"][i]
            length = data["response_lengths"][i]
            start = len(source) - length
            end = len(source)
            # Only the final target can differ. Its predicting hidden state
            # precedes it, so the original terminal target needs no branch
            # forward even if the later prompt inserts a closing think tag.
            if end > len(tokens) or source[:-1] != tokens[: end - 1]:
                raise ValueError(
                    f"Trajectory {sample_index} turn {data['turn_indices'][i]} has a changed scored prefix; "
                    "use exact token history before enabling trajectory packing"
                )
            lo, hi = start - prompt_length, end - prompt_length
            if any(mask[lo:hi]):
                raise ValueError(f"Overlapping response spans in trajectory {sample_index}")
            targets[start:end] = source[start:end]
            mask[lo:hi] = turn_mask
            token_rewards[lo:hi] = [data["rewards"][i]] * length
            for key, value in fields.items():
                value[lo:hi] = data[key][i]

        if "rollout_log_probs" in fields:
            # The actor's CP slicer accepts lists or tensors, not numpy.
            fields["rollout_log_probs"] = fields["rollout_log_probs"].tolist()
        row = {key: data[key][last] for key in data}
        row.update(
            tokens=tokens,
            target_tokens=targets,
            response_lengths=response_length,
            loss_masks=mask,
            token_rewards=token_rewards,
            # Match loss_function's original clamp PER TURN, including dummy
            # and filtered turns, rather than clamping the merged mask once.
            loss_normalization_counts=sum(max(sum(data["loss_masks"][i]), 1) for i in positions),
            turn_indices=0,
            packed_turn_metrics={key: [data[key][i] for i in positions] for key in metric_fields if key in data},
            **fields,
        )
        for key in metric_fields:
            if key in row and key != "turn_indices":
                row[key] = sum(data[key][i] for i in positions) / len(positions)
        for key in packed:
            packed[key].append(row[key])
    return packed
