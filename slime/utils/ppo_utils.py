# Adapt from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/models/utils.py
# and https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ppo_utils/experience_maker.py

from argparse import Namespace

import torch
import torch.distributed as dist
import torch.nn.functional as F


@torch.compile(dynamic=True)
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_loss_type: str,
    importance_ratio: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        kl_loss_type: Type of KL estimator (k1, k2, k3, low_var_kl).
        importance_ratio: Optional IS ratio (π_θ/π_old) for unbiased KL estimation.
    """
    log_ratio = log_probs.float() - log_probs_base.float()

    if kl_loss_type == "k1":
        kl = log_ratio
    elif kl_loss_type == "k2":
        kl = log_ratio**2 / 2.0
    elif kl_loss_type in ["k3", "low_var_kl"]:
        # The non negative kl approximation in
        # http://joschu.net/blog/kl-approx.html
        # Besides non negative, it is also unbiased and have lower variance.
        log_ratio = -log_ratio
        kl = log_ratio.exp() - 1 - log_ratio
    else:
        raise ValueError(f"Unknown kl_loss_type: {kl_loss_type}")

    # Apply IS ratio for unbiased KL estimation (DeepSeek-V3.2)
    if importance_ratio is not None:
        kl = importance_ratio * kl

    # Clamp only for low_var_kl for numerical stability
    if kl_loss_type == "low_var_kl":
        kl = torch.clamp(kl, min=-10, max=10)

    return kl


def compute_opsm_mask(
    args: Namespace,
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    advantages: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Off-Policy Sequence Masking (OPSM) mask.

    Args:
        args: Configuration containing `opsm_delta` threshold.
        full_log_probs: Current policy log-probs per sample.
        full_old_log_probs: Old policy log-probs per sample.
        advantages: Advantage values per sample.
        loss_masks: Loss masks per sample.

    Returns:
        Tuple of `(opsm_mask, opsm_clipfrac)` where `opsm_mask` is a
        concatenated tensor of per-token masks and
        `opsm_clipfrac` is the count of masked sequences.
    """
    opsm_mask_list = []
    device = advantages[0].device
    opsm_clipfrac = torch.tensor(0.0, device=device)

    for full_log_prob, full_old_log_prob, advantage, loss_mask in zip(
        full_log_probs, full_old_log_probs, advantages, loss_masks, strict=False
    ):
        # Calculate sequence-level KL
        seq_kl = ((full_old_log_prob - full_log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)

        # Create mask: 0 if (advantage < 0 and seq_kl > delta), else 1
        mask = ((advantage < 0) & (seq_kl > args.opsm_delta)).float()
        opsm_clipfrac += mask.sum() / torch.clamp_min(loss_mask.sum(), 1)

        opsm_mask_list.append(1 - mask)

    opsm_mask = torch.cat(opsm_mask_list, dim=0)
    return opsm_mask, opsm_clipfrac


def compute_gspo_kl(
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    local_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> torch.Tensor:
    """Compute GSPO-style per-sequence KL divergence.

    Args:
        full_log_probs: Current policy log-probs per sample (full or CP-local).
        full_old_log_probs: Old policy log-probs per sample (full or CP-local).
        local_log_probs: Local (CP-local) log-probs for expansion shape reference.
        loss_masks: Loss masks per sample.

    Returns:
        Concatenated tensor of per-token KL values where each token in a
        sequence has the same KL value (the sequence-level KL).
    """
    # Compute sequence-level KL and expand to per-token
    ppo_kl = [
        ((old_logprob - log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)
        for log_prob, old_logprob, loss_mask in zip(full_log_probs, full_old_log_probs, loss_masks, strict=False)
    ]
    ppo_kl = [kl.expand_as(log_prob) for kl, log_prob in zip(ppo_kl, local_log_probs, strict=False)]
    ppo_kl = torch.cat(ppo_kl, dim=0)

    return ppo_kl


def compute_sequence_log_ratio(
    full_log_probs: list[torch.Tensor],
    full_rollout_log_probs: list[torch.Tensor],
    local_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> torch.Tensor:
    """Return each response's mean log-ratio expanded over its local tokens."""
    sequence_log_ratios = [
        (((log_prob.detach() - rollout_log_prob.detach()) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1))
        for log_prob, rollout_log_prob, loss_mask in zip(
            full_log_probs, full_rollout_log_probs, loss_masks, strict=True
        )
    ]
    return torch.cat(
        [ratio.expand_as(log_prob) for ratio, log_prob in zip(sequence_log_ratios, local_log_probs, strict=True)],
        dim=0,
    )


@torch.compile(dynamic=True)
def compute_policy_loss(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
):
    ratio = (-ppo_kl).exp()
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    clipfrac = torch.gt(pg_losses2, pg_losses1).float()

    if eps_clip_c is not None:
        assert (
            eps_clip_c > 1.0
        ), f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {eps_clip_c}."
        pg_losses3 = -eps_clip_c * advantages
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    else:
        pg_losses = clip_pg_losses1

    return pg_losses, clipfrac


def compute_dis_policy_loss(
    log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    log_ratio: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute Direct Double-Sided Importance Sampling (DIS).

    DIS uses the rollout policy directly as the behavior policy,

        r = exp(log pi_train - log pi_rollout),

    and removes every token outside ``(1 - eps_clip, 1 + eps_clip_high)``
    from the policy gradient.  Unlike PPO clipping and the DPPO masks, the
    double-sided gate is independent of the advantage sign.  The importance
    weight and gate are detached, matching the paper's score-function form
    ``f(r) * A * log pi_train``; gradients flow only through ``log_probs``.
    """
    tensors = {
        "log_probs": log_probs,
        "rollout_log_probs": rollout_log_probs,
        "advantages": advantages,
    }
    if log_ratio is not None:
        tensors["log_ratio"] = log_ratio
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
        if tensor.device != log_probs.device:
            raise ValueError(f"{name} must be on {log_probs.device}, got {tensor.device}")
    if log_probs.shape != rollout_log_probs.shape or log_probs.shape != advantages.shape:
        raise ValueError(
            "log_probs, rollout_log_probs, and advantages must have identical shapes; "
            f"got {tuple(log_probs.shape)}, {tuple(rollout_log_probs.shape)}, "
            f"and {tuple(advantages.shape)}"
        )
    if log_ratio is not None and log_ratio.shape != log_probs.shape:
        raise ValueError(
            f"log_ratio must match log_probs shape; got {tuple(log_ratio.shape)} and {tuple(log_probs.shape)}"
        )

    lower = 1.0 - float(eps_clip)
    upper = 1.0 + float(eps_clip_high)
    if not (0.0 < lower < 1.0):
        raise ValueError(f"DIS requires 0 < 1 - eps_clip < 1, got eps_clip={eps_clip!r}")
    if not torch.isfinite(torch.tensor(upper)) or upper <= 1.0:
        raise ValueError(f"DIS requires a finite eps_clip_high > 0, got {eps_clip_high!r}")

    calc_dtype = torch.float64 if log_probs.dtype == torch.float64 else torch.float32
    with torch.no_grad():
        if log_ratio is None:
            log_ratio = log_probs.detach().to(calc_dtype) - rollout_log_probs.detach().to(calc_dtype)
        else:
            log_ratio = log_ratio.detach().to(calc_dtype)
        detached_advantages = advantages.detach().to(calc_dtype)
        finite_values = torch.isfinite(log_ratio).all() & torch.isfinite(detached_advantages).all()
        if finite_values.device.type == "cuda":
            torch._assert_async(finite_values, "DIS log-probs and advantages must contain only finite values")
        elif not bool(finite_values):
            raise ValueError("DIS log-probs and advantages must contain only finite values")

        log_lower = torch.tensor(lower, dtype=calc_dtype, device=log_probs.device).log()
        log_upper = torch.tensor(upper, dtype=calc_dtype, device=log_probs.device).log()
        below = log_ratio <= log_lower
        above = log_ratio >= log_upper
        valid = ~(below | above)

        # Keep the raw importance ratio observable independently of the DIS
        # gate/weight.  Clamp only for finite diagnostics: the actual gate is
        # still decided from the unclamped log-ratio above.
        importance_ratio = log_ratio.clamp(min=-20.0, max=20.0).exp()

        # The ratio only contributes for valid tokens.  Clamping before exp
        # prevents an out-of-range token from producing inf that later meets a
        # zero mask; values inside the open interval remain exact.
        ratio = log_ratio.clamp(min=log_lower, max=log_upper).exp()
        importance_weight = torch.where(valid, ratio, torch.zeros_like(ratio))

    loss_dtype = log_probs.dtype
    weight_for_loss = importance_weight.to(loss_dtype)
    advantages_for_loss = advantages.detach().to(loss_dtype)
    pg_losses = -advantages_for_loss * weight_for_loss * log_probs
    return {
        "pg_losses": pg_losses,
        "pg_clipfrac": (~valid).to(loss_dtype),
        "pg_upper_clipfrac": above.to(loss_dtype),
        "pg_lower_clipfrac": below.to(loss_dtype),
        "dis_importance_ratio": importance_ratio.to(loss_dtype),
        "dis_importance_weight": importance_weight.to(loss_dtype),
        "dis_valid_token_frac": valid.to(loss_dtype),
    }


def compute_ppo_clip_diagnostics(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
) -> dict[str, torch.Tensor]:
    """Return sign-resolved ordinary-PPO clipping indicators.

    ``compute_policy_loss`` reports the union of the standard upper/lower
    clipping events.  Keeping their advantage-sign decomposition observable is
    important when a cross-engine old-policy anchor clips before any optimizer
    update.  Dual clip is a separate negative-advantage high-ratio event and is
    emitted only when configured.
    """
    ratio = (-ppo_kl.detach()).exp()
    detached_advantages = advantages.detach()
    positive_advantage = detached_advantages > 0
    negative_advantage = detached_advantages < 0
    diagnostics = {
        "pg_upper_clipfrac": (positive_advantage & (ratio > 1 + eps_clip_high)).float(),
        "pg_lower_clipfrac": (negative_advantage & (ratio < 1 - eps_clip)).float(),
    }
    if eps_clip_c is not None:
        diagnostics["pg_dual_clipfrac"] = (negative_advantage & (ratio > eps_clip_c)).float()
    return diagnostics


def compute_dppo_binary_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    loss_mode: str,
    eps_clip_c: float | None = None,
):
    """Compute DPPO binary-TV/KL policy loss against the configured old policy anchor.

    Binary variants split the vocabulary into the sampled token and all other
    tokens, then mask token updates that move outside the divergence threshold
    in the advantage-driven direction.
    """
    prob = log_probs.exp()
    old_prob = old_log_probs.exp()

    if loss_mode == "dppo_binary_tv":
        invalid_positive_mask = (prob - old_prob) > eps_clip_high
        invalid_negative_mask = (prob - old_prob) < -eps_clip
    elif loss_mode == "dppo_binary_kl":
        prob_for_other = torch.clamp(1.0 - prob, min=1e-8)
        old_prob_for_other = torch.clamp(1.0 - old_prob, min=1e-8)
        binary_kl = old_prob * (old_log_probs - log_probs) + old_prob_for_other * torch.log(
            old_prob_for_other / prob_for_other
        )
        invalid_positive_mask = (binary_kl > eps_clip_high) & (prob > old_prob)
        invalid_negative_mask = (binary_kl > eps_clip) & (prob < old_prob)
    else:
        raise ValueError(f"Unsupported DPPO loss mode: {loss_mode}")

    positive_advantage_mask = advantages > 0
    invalid_mask = torch.where(positive_advantage_mask, invalid_positive_mask, invalid_negative_mask)
    valid_mask = 1.0 - invalid_mask.detach().float()

    ratio_clip_c = 20.0 if eps_clip_c is None else eps_clip_c
    ratio = torch.exp(torch.clamp(log_probs - old_log_probs, min=-20.0, max=20.0))
    ratio = torch.clamp(ratio, max=ratio_clip_c).detach()

    pg_losses = -advantages * ratio * valid_mask * log_probs
    upper_clipfrac = (positive_advantage_mask & invalid_positive_mask).float()
    lower_clipfrac = (~positive_advantage_mask & invalid_negative_mask).float()
    clipfrac = invalid_mask.float()
    return {
        "pg_losses": pg_losses,
        "pg_clipfrac": clipfrac,
        "pg_upper_clipfrac": upper_clipfrac,
        "pg_lower_clipfrac": lower_clipfrac,
    }


def compute_dppo_predictive_topk_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    behavior_support_log_probs: torch.Tensor,
    current_support_log_probs: torch.Tensor,
    support_valid_mask: torch.Tensor,
    advantages: torch.Tensor,
    delta: float,
    tail_estimator: str,
    vocab_size: int,
    eps_clip_c: float | None = 5.0,
):
    """Compute DPPO's predictive Top-K-KL masked policy loss.

    ``behavior_support_log_probs`` and ``current_support_log_probs`` describe
    the same retained support ``S = TopK(mu, K) union {sampled token}``.  The
    last axis is the support axis and ``support_valid_mask`` excludes padding.
    The retained support plus one residual-tail bucket defines

        D = sum_{i in S} mu_i (log(mu_i) - log(pi_i))
            + mu_tail log(mu_tail / pi_tail).

    The predictive coefficient is the directional derivative of that
    divergence along the sampled-token policy-gradient direction:

        dot_D = (pi_k - mu_k) + sum_{i in S} pi_i (mu_i - pi_i)
                + tail_term.

    ``tail_term`` is either the aggregated-tail estimate
    ``pi_tail * (mu_tail - pi_tail)`` or the uniform-tail estimate obtained by
    dividing it by ``vocab_size - |S|``.  A token is masked exactly when it is
    outside the KL trust region and its advantage-scaled ``dot_D`` is positive.

    All quantities used to form the importance ratio and predictive mask are
    detached.  Consequently, the returned loss has a gradient only through
    the explicit sampled-token ``log_probs`` factor.
    """

    tensors = {
        "log_probs": log_probs,
        "old_log_probs": old_log_probs,
        "behavior_support_log_probs": behavior_support_log_probs,
        "current_support_log_probs": current_support_log_probs,
        "support_valid_mask": support_valid_mask,
        "advantages": advantages,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")

    if log_probs.shape != old_log_probs.shape or log_probs.shape != advantages.shape:
        raise ValueError(
            "log_probs, old_log_probs, and advantages must have identical shapes; "
            f"got {tuple(log_probs.shape)}, {tuple(old_log_probs.shape)}, and {tuple(advantages.shape)}"
        )
    if behavior_support_log_probs.shape != current_support_log_probs.shape:
        raise ValueError(
            "behavior and current support log-probs must have identical shapes; "
            f"got {tuple(behavior_support_log_probs.shape)} and {tuple(current_support_log_probs.shape)}"
        )
    if support_valid_mask.shape != behavior_support_log_probs.shape:
        raise ValueError(
            "support_valid_mask must have the same shape as support log-probs; "
            f"got {tuple(support_valid_mask.shape)} and {tuple(behavior_support_log_probs.shape)}"
        )
    if behavior_support_log_probs.ndim != log_probs.ndim + 1:
        raise ValueError(
            "support log-probs must add exactly one trailing support dimension to sampled log-probs; "
            f"got {tuple(behavior_support_log_probs.shape)} versus {tuple(log_probs.shape)}"
        )
    if behavior_support_log_probs.shape[:-1] != log_probs.shape:
        raise ValueError(
            "support log-prob leading dimensions must match sampled log-probs; "
            f"got {tuple(behavior_support_log_probs.shape[:-1])} versus {tuple(log_probs.shape)}"
        )
    if behavior_support_log_probs.shape[-1] == 0:
        raise ValueError("the retained support dimension must be non-empty")
    if support_valid_mask.dtype != torch.bool:
        raise TypeError(f"support_valid_mask must have dtype torch.bool, got {support_valid_mask.dtype}")

    floating_tensors = {name: tensor for name, tensor in tensors.items() if name != "support_valid_mask"}
    for name, tensor in floating_tensors.items():
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
        if tensor.device != log_probs.device:
            raise ValueError(f"{name} must be on {log_probs.device}, got {tensor.device}")
    if support_valid_mask.device != log_probs.device:
        raise ValueError(f"support_valid_mask must be on {log_probs.device}, got {support_valid_mask.device}")

    if tail_estimator not in {"aggregated", "uniform"}:
        raise ValueError(f"tail_estimator must be 'aggregated' or 'uniform', got {tail_estimator!r}")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError(f"vocab_size must be a positive integer, got {vocab_size!r}")
    if not isinstance(delta, (int, float)) or not torch.isfinite(torch.tensor(float(delta))) or delta < 0:
        raise ValueError(f"delta must be a finite non-negative number, got {delta!r}")

    ratio_clip_c = 5.0 if eps_clip_c is None else eps_clip_c
    if (
        isinstance(ratio_clip_c, bool)
        or not isinstance(ratio_clip_c, (int, float))
        or not torch.isfinite(torch.tensor(float(ratio_clip_c)))
        or ratio_clip_c <= 0
    ):
        raise ValueError(f"eps_clip_c must be a finite positive number, got {eps_clip_c!r}")

    # Top-K statistics may be stored in fp16/bf16.  Use fp32 for the
    # probability sums and KL unless the caller explicitly supplies fp64.
    calc_dtype = torch.float64 if any(t.dtype == torch.float64 for t in floating_tensors.values()) else torch.float32
    probability_tolerance = 1e-5

    def require_tensor_condition(condition: torch.Tensor, message: str) -> None:
        """Validate values without forcing a CUDA host synchronization."""
        if condition.device.type == "cuda":
            torch._assert_async(condition, message)
        elif not bool(condition):
            raise ValueError(message)

    with torch.no_grad():
        sampled_log_prob = log_probs.detach().to(calc_dtype)
        sampled_old_log_prob = old_log_probs.detach().to(calc_dtype)
        detached_advantages = advantages.detach().to(calc_dtype)
        valid = support_valid_mask.detach()
        behavior_support_logs = torch.where(
            valid,
            behavior_support_log_probs.detach().to(calc_dtype),
            torch.zeros((), device=log_probs.device, dtype=calc_dtype),
        )
        current_support_logs = torch.where(
            valid,
            current_support_log_probs.detach().to(calc_dtype),
            torch.zeros((), device=log_probs.device, dtype=calc_dtype),
        )

        valid_sampled_values = (
            torch.isfinite(sampled_log_prob).all()
            & torch.isfinite(sampled_old_log_prob).all()
            & torch.isfinite(detached_advantages).all()
        )
        valid_support_values = torch.isfinite(behavior_support_logs).all() & torch.isfinite(current_support_logs).all()
        require_tensor_condition(
            valid_sampled_values,
            "sampled log-probs and advantages must contain only finite values",
        )
        require_tensor_condition(valid_support_values, "valid support log-probs must contain only finite values")

        behavior_support_probs = behavior_support_logs.exp() * valid
        current_support_probs = current_support_logs.exp() * valid
        behavior_support_mass = behavior_support_probs.sum(dim=-1)
        current_support_mass = current_support_probs.sum(dim=-1)
        support_size = valid.sum(dim=-1)

        require_tensor_condition((support_size <= vocab_size).all(), "retained-support size cannot exceed vocab_size")
        if tail_estimator == "uniform":
            require_tensor_condition(
                (support_size < vocab_size).all(),
                "uniform-tail estimator requires vocab_size - retained_support_size > 0",
            )
        require_tensor_condition(
            (behavior_support_mass <= 1.0 + probability_tolerance).all(),
            "behavior retained-support probabilities sum to more than one",
        )
        require_tensor_condition(
            (current_support_mass <= 1.0 + probability_tolerance).all(),
            "current retained-support probabilities sum to more than one",
        )

        sampled_behavior_prob = sampled_old_log_prob.exp()
        sampled_current_prob = sampled_log_prob.exp()
        require_tensor_condition(
            (sampled_behavior_prob <= 1.0 + probability_tolerance).all(),
            "sampled old log-probs encode probabilities greater than one",
        )
        require_tensor_condition(
            (sampled_current_prob <= 1.0 + probability_tolerance).all(),
            "sampled current log-probs encode probabilities greater than one",
        )

        behavior_tail_mass = (1.0 - behavior_support_mass).clamp_min(0.0)
        current_tail_mass = (1.0 - current_support_mass).clamp_min(0.0)
        min_positive = torch.finfo(calc_dtype).tiny

        retained_kl = (behavior_support_probs * (behavior_support_logs - current_support_logs) * valid).sum(dim=-1)
        tail_kl = torch.where(
            behavior_tail_mass > 0,
            behavior_tail_mass
            * (behavior_tail_mass.clamp_min(min_positive).log() - current_tail_mass.clamp_min(min_positive).log()),
            torch.zeros_like(behavior_tail_mass),
        )
        has_support = support_size > 0
        topk_kl = torch.where(has_support, (retained_kl + tail_kl).clamp_min(0.0), 0.0)

        local_term = torch.where(has_support, sampled_current_prob - sampled_behavior_prob, 0.0)
        retained_term = (current_support_probs * (behavior_support_probs - current_support_probs) * valid).sum(dim=-1)
        aggregated_tail_term = current_tail_mass * (behavior_tail_mass - current_tail_mass)
        if tail_estimator == "uniform":
            tail_denominator = (vocab_size - support_size).to(calc_dtype)
            tail_term = aggregated_tail_term / tail_denominator
        else:
            tail_term = aggregated_tail_term
        predictive_dot = torch.where(has_support, local_term + retained_term + tail_term, 0.0)

        outside = topk_kl > float(delta)
        predictive_increasing = detached_advantages * predictive_dot > 0
        ratio_increasing = detached_advantages * local_term > 0
        direction_disagreement = predictive_increasing != ratio_increasing
        invalid_mask = outside & predictive_increasing
        positive_advantage = detached_advantages > 0
        negative_advantage = detached_advantages < 0
        zero_advantage = detached_advantages == 0

        # Keep the raw sampled-token ratio observable independently of the
        # cap used by the predictive-DPPO loss.  Both tensors are detached and
        # linearly reducible, so logging them cannot alter the objective.
        importance_ratio = (sampled_log_prob - sampled_old_log_prob).clamp(min=-20.0, max=20.0).exp()
        importance_weight = importance_ratio.clamp(max=float(ratio_clip_c))
        valid_loss_mask = (~invalid_mask).to(calc_dtype)

        # Diagnostic sufficient statistics deliberately remain per-token and
        # linearly reducible.  In particular, do not form conditional rates or
        # mass ratios here: a microbatch can contain only one advantage sign,
        # and averaging microbatch-local ratios would bias the step metric.
        positive_clipped = invalid_mask & positive_advantage
        negative_clipped = invalid_mask & negative_advantage
        positive_kept = (~invalid_mask) & positive_advantage
        negative_kept = (~invalid_mask) & negative_advantage
        diagnostic_categories = {
            "positive_clipped": positive_clipped.to(calc_dtype),
            "positive_kept": positive_kept.to(calc_dtype),
            "negative_clipped": negative_clipped.to(calc_dtype),
            "negative_kept": negative_kept.to(calc_dtype),
        }

        update_mass = detached_advantages.abs() * importance_weight
        signed_update = detached_advantages * importance_weight
        kept_update_mass = update_mass * valid_loss_mask
        kept_signed_update = signed_update * valid_loss_mask
        sampled_logit_first_order_unmasked = signed_update * (1.0 - sampled_current_prob)
        sampled_logit_first_order_kept = sampled_logit_first_order_unmasked * valid_loss_mask

        # A row-wise additive shift in logits cancels from log-probabilities.
        # Centering each retained support therefore recovers exactly the
        # identifiable part of the logits without storing [token, vocab].
        support_denominator = support_size.clamp_min(1).to(calc_dtype)
        behavior_support_mean = (behavior_support_logs * valid).sum(dim=-1) / support_denominator
        current_support_mean = (current_support_logs * valid).sum(dim=-1) / support_denominator
        behavior_centered = torch.where(
            valid,
            behavior_support_logs - behavior_support_mean.unsqueeze(-1),
            torch.zeros((), dtype=calc_dtype, device=log_probs.device),
        )
        current_centered = torch.where(
            valid,
            current_support_logs - current_support_mean.unsqueeze(-1),
            torch.zeros((), dtype=calc_dtype, device=log_probs.device),
        )
        behavior_centered_std = torch.where(
            has_support,
            ((behavior_centered.square() * valid).sum(dim=-1) / support_denominator).sqrt(),
            0.0,
        )
        current_centered_std = torch.where(
            has_support,
            ((current_centered.square() * valid).sum(dim=-1) / support_denominator).sqrt(),
            0.0,
        )
        centered_logit_abs_diff = torch.where(
            has_support,
            ((current_centered - behavior_centered).abs() * valid).sum(dim=-1) / support_denominator,
            0.0,
        )
        behavior_sampled_support_gap = torch.where(
            has_support,
            sampled_old_log_prob - behavior_support_mean,
            0.0,
        )
        current_sampled_support_gap = torch.where(
            has_support,
            sampled_log_prob - current_support_mean,
            0.0,
        )

        if behavior_support_logs.size(-1) >= 2:
            negative_infinity = torch.tensor(float("-inf"), dtype=calc_dtype, device=log_probs.device)
            behavior_top2 = behavior_support_logs.masked_fill(~valid, negative_infinity).topk(2, dim=-1).values
            current_top2 = current_support_logs.masked_fill(~valid, negative_infinity).topk(2, dim=-1).values
            has_two_support_tokens = support_size >= 2
            behavior_top1_top2_margin = torch.where(
                has_two_support_tokens,
                behavior_top2[..., 0] - behavior_top2[..., 1],
                0.0,
            )
            current_top1_top2_margin = torch.where(
                has_two_support_tokens,
                current_top2[..., 0] - current_top2[..., 1],
                0.0,
            )
        else:
            behavior_top1_top2_margin = torch.zeros_like(sampled_old_log_prob)
            current_top1_top2_margin = torch.zeros_like(sampled_log_prob)

        dppo_diagnostic_metrics = {
            "dppo/adv_positive_token_frac": positive_advantage.to(calc_dtype),
            "dppo/adv_negative_token_frac": negative_advantage.to(calc_dtype),
            "dppo/adv_zero_token_frac": zero_advantage.to(calc_dtype),
            "dppo/upper_clip_joint_frac": diagnostic_categories["positive_clipped"],
            "dppo/lower_clip_joint_frac": diagnostic_categories["negative_clipped"],
            "dppo/positive_kept_joint_frac": diagnostic_categories["positive_kept"],
            "dppo/negative_kept_joint_frac": diagnostic_categories["negative_kept"],
            "dppo/update_mass_positive_mean": update_mass * positive_advantage,
            "dppo/update_mass_negative_mean": update_mass * negative_advantage,
            "dppo/masked_update_mass_positive_mean": update_mass * positive_clipped,
            "dppo/masked_update_mass_negative_mean": update_mass * negative_clipped,
            "dppo/kept_update_mass_mean": kept_update_mass,
            "dppo/net_logprob_push_unmasked_numerator_mean": signed_update,
            "dppo/net_logprob_push_kept_numerator_mean": kept_signed_update,
            # These three signed moments intentionally use the same original
            # active-token denominator after the loss path's DP×CP reduction.
            # Unlike ``net_logprob_push_*``, no absolute-mass normalization is
            # applied, so the mask delta is directly additive and comparable.
            "dppo/signed_update_unmasked": signed_update,
            "dppo/signed_update_kept": kept_signed_update,
            "dppo/signed_update_mask_delta": kept_signed_update - signed_update,
            # For an optimizer-ascent coefficient c=A*pi/mu, the sampled-logit
            # direction is exactly c*(1-pi_k).  These are observational
            # logit-space moments; they do not alter the loss or its gradient.
            "dppo/sampled_logit_first_order_unmasked": sampled_logit_first_order_unmasked,
            "dppo/sampled_logit_first_order_kept": sampled_logit_first_order_kept,
            "dppo/rollout_predictive_support_centered_logit_std": behavior_centered_std,
            "dppo/train_predictive_support_centered_logit_std": current_centered_std,
            "dppo/train_rollout_predictive_support_centered_logit_abs_diff": centered_logit_abs_diff,
            "dppo/rollout_sampled_to_support_mean_logit_gap": behavior_sampled_support_gap,
            "dppo/train_sampled_to_support_mean_logit_gap": current_sampled_support_gap,
            "dppo/train_rollout_sampled_to_support_gap_delta": (
                current_sampled_support_gap - behavior_sampled_support_gap
            ),
            "dppo/rollout_support_top1_top2_logit_margin": behavior_top1_top2_margin,
            "dppo/train_support_top1_top2_logit_margin": current_top1_top2_margin,
            "dppo/train_rollout_support_top1_top2_margin_delta": (
                current_top1_top2_margin - behavior_top1_top2_margin
            ),
        }
        for category, indicator in diagnostic_categories.items():
            dppo_diagnostic_metrics[f"dppo/train_sampled_prob_{category}_joint_mean"] = (
                sampled_current_prob * indicator
            )
            dppo_diagnostic_metrics[f"dppo/rollout_sampled_prob_{category}_joint_mean"] = (
                sampled_behavior_prob * indicator
            )

    # Do not use sampled_log_prob here: it is detached.  This explicit factor
    # is the sole autograd path by construction.
    pg_losses = -detached_advantages * importance_weight * valid_loss_mask * log_probs
    return {
        "pg_losses": pg_losses,
        "pg_clipfrac": invalid_mask.to(calc_dtype),
        "pg_upper_clipfrac": (invalid_mask & positive_advantage).to(calc_dtype),
        "pg_lower_clipfrac": (invalid_mask & negative_advantage).to(calc_dtype),
        "dppo_importance_ratio": importance_ratio,
        "dppo_importance_weight": importance_weight,
        "dppo_topk_kl": topk_kl,
        "dppo_predictive_dot": predictive_dot,
        "dppo_outside": outside.to(calc_dtype),
        "dppo_predictive_increasing": predictive_increasing.to(calc_dtype),
        "dppo_ratio_increasing": ratio_increasing.to(calc_dtype),
        "dppo_direction_disagreement": direction_disagreement.to(calc_dtype),
        "dppo_behavior_tail_mass": behavior_tail_mass,
        "dppo_current_tail_mass": current_tail_mass,
        "dppo_predictive_tail_term": tail_term,
        **dppo_diagnostic_metrics,
    }


def compute_log_probs(logits: torch.Tensor, tokens: torch.Tensor, process_group: dist.ProcessGroup | None):
    # TODO: when megatron is not installed, fall back to naive implementation
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

    # convert to [seq_len, batch_size, vocab_size] as expected by fused_vocab_parallel_cross_entropy
    logits = logits.unsqueeze(1)
    tokens = tokens.unsqueeze(1)
    return -fused_vocab_parallel_cross_entropy(logits, tokens, process_group)


# from https://github.com/volcengine/verl/blob/0bdf7f469854815177e73dcfe9e420836c952e6e/verl/utils/megatron/tensor_parallel.py#L99
class _VocabParallelEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor, process_group: dist.ProcessGroup) -> torch.Tensor:

        @torch.compile(dynamic=True)
        def mul_reduce(a, b):
            return (a * b).sum(dim=-1, keepdim=True)

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=process_group)
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        dist.all_reduce(sum_softmax_times_logits, group=process_group)
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # reuse softmax_logits as grad
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        softmax_logits.mul_(vocab_parallel_logits)
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        # recover vocab_parallel_logits
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        softmax_logits.mul_(-1)
        return softmax_logits, None


def compute_entropy_from_logits(logits: torch.Tensor, process_group) -> torch.Tensor:
    return _VocabParallelEntropy.apply(logits, process_group)


class _VocabParallelEntropyWithDirectionalMoment(torch.autograd.Function):
    """Compute entropy plus the full-vocabulary DPPO directional moment.

    For probabilities ``p=softmax(z)`` and entropy ``H``, the part of the
    entropy derivative shared by every sampled-token policy-gradient direction
    is

        M = sum_i p_i^2 (log(p_i) + H)
          = sum_i p_i^2 (z_i - E_p[z]).

    Returning ``M`` from the same softmax pass avoids a second full-vocabulary
    scan solely for diagnostics.  ``M`` is explicitly non-differentiable; the
    entropy output keeps the exact backward used by ``_VocabParallelEntropy``.
    """

    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor, process_group):

        @torch.compile(dynamic=True)
        def mul_reduce(a, b):
            return (a * b).sum(dim=-1, keepdim=True)

        @torch.compile(dynamic=True)
        def directional_moment(probabilities, logits, expected_logit):
            # Inductor fuses the elementwise expression, so this adds a
            # reduction without retaining another [tokens, vocab] tensor.
            return (probabilities.square() * (logits - expected_logit)).sum(dim=-1, keepdim=True)

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=process_group)
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        dist.all_reduce(sum_softmax_times_logits, group=process_group)
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits

        moment = directional_moment(softmax_logits, vocab_parallel_logits, sum_softmax_times_logits)
        dist.all_reduce(moment, group=process_group)

        entropy = entropy.squeeze(dim=-1)
        moment = moment.squeeze(dim=-1)
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        ctx.mark_non_differentiable(moment)
        return entropy, moment

    @staticmethod
    def backward(ctx, grad_entropy: torch.Tensor, _grad_moment: torch.Tensor):
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # Match _VocabParallelEntropy.backward exactly.  The input is a clone
        # owned by the entropy path, so the temporary in-place centering is safe.
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        softmax_logits.mul_(vocab_parallel_logits)
        softmax_logits.mul_(grad_entropy.unsqueeze(dim=-1))
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        softmax_logits.mul_(-1)
        return softmax_logits, None


def compute_entropy_and_dppo_directional_moment_from_logits(
    logits: torch.Tensor,
    process_group,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return entropy and ``sum p^2(log p + H)`` from one TP softmax pass."""
    return _VocabParallelEntropyWithDirectionalMoment.apply(logits, process_group)


def get_grpo_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
):
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns


def get_rloo_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
    n_samples_per_prompt: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Compute outcome-level RLOO advantages grouped by prompt.
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Each contiguous group of ``n_samples_per_prompt`` samples is treated as one
    prompt group. The leave-one-out baseline for a sample is the mean reward of
    the other samples in that group. For singleton groups, no baseline is used.
    """
    assert n_samples_per_prompt >= 1, "n_samples_per_prompt must be >= 1 for RLOO"

    advantages = torch.empty_like(rewards)
    for start in range(0, rewards.numel(), n_samples_per_prompt):
        end = min(start + n_samples_per_prompt, rewards.numel())
        group_rewards = rewards[start:end]
        group_size = group_rewards.numel()
        if group_size == 1:
            group_advantages = group_rewards
        else:
            group_sum = group_rewards.sum()
            loo_baseline = (group_sum - group_rewards) / (group_size - 1)
            group_advantages = group_rewards - loo_baseline
        advantages[start:end] = group_advantages

    returns = get_grpo_returns(rewards, kl)
    return get_grpo_returns(advantages, kl), returns


def get_reinforce_plus_plus_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    response_lengths: list[int],
    total_lengths: list[int],
    kl_coef: float,
    gamma: float,
) -> list[torch.Tensor]:
    """
    Calculates discounted returns for REINFORCE++ (https://arxiv.org/pdf/2501.03262)

    Args:
        rewards (Tensor): A tensor of scalar rewards for each sequence.
        kl (List[Tensor]): List of per-token KL divergence tensors for sequence chunks.
        loss_masks (List[Tensor]): List of response-only loss masks for each full sequence.
        response_lengths (List[int]): The full length of each response sequence.
        total_lengths (List[int]): The full length of each sequence (prompt + response).
        kl_coef (float): Coefficient for the KL penalty.
        gamma (float): The discount factor.

    Returns:
        List[torch.Tensor]: A list of return (G_t) tensors for the
                            local sequence chunks owned by the current GPU rank.
    """
    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()

    final_returns_chunks = []
    for i in range(len(rewards)):
        local_kl_chunk = kl[i]
        total_len, response_len = total_lengths[i], response_lengths[i]

        if cp_size > 1:
            # Step 1,2:Gather all chunks and token_offsets from all ranks and reconstruct the full response tensor by splitting and placing each part
            from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

            full_kl_response = all_gather_with_cp(local_kl_chunk, total_len, response_len)
        else:
            full_kl_response = local_kl_chunk

        # Step 3: Compute returns on full response kl tensor.
        full_mask = loss_masks[i]
        assert full_mask.sum().item() > 0, f"Sequence at index {i} is fully masked."
        masked_kl = full_kl_response * full_mask
        token_level_rewards = -kl_coef * masked_kl
        last_idx = full_mask.nonzero(as_tuple=True)[0][-1]
        token_level_rewards[last_idx] += rewards[i]

        returns_for_seq = torch.zeros_like(token_level_rewards)
        running_return = 0.0
        for t in reversed(range(token_level_rewards.size(0))):
            # G_t = r_t + gamma * G_{t+1}
            running_return = token_level_rewards[t] + gamma * running_return
            returns_for_seq[t] = running_return

        # Step 4: Pick up the results corresponding to our local chunk's parts.
        if cp_size > 1:
            from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

            local_returns_chunk = slice_log_prob_with_cp(returns_for_seq, total_len, response_len)
        else:
            local_returns_chunk = returns_for_seq

        final_returns_chunks.append(local_returns_chunk)

    return final_returns_chunks


def get_reinforce_plus_plus_baseline_advantages(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    kl_coef: float,
) -> list[torch.Tensor]:
    """
    Calculates the unwhitened advantages for the REINFORCE++-baseline algorithm.
    Broadcasting the scalar (reward - group_baseline) to each token.

    Args:
        rewards (Tensor): A tensor of scalar rewards, where the group-wise
                                baseline has already been subtracted.
        kl (list[Tensor]): A list of per-token KL divergence tensors. Used to
                                 get the shape for broadcasting.
        loss_masks (list[Tensor]): A list of per-token loss masks.
        kl_coef (float): Coefficient for the KL penalty.

    Returns:
        list[Tensor]: A list of tensors containing the unwhitened advantages.
    """
    # Broadcast to get unwhitened advantages
    unwhitened_advantages = [
        torch.ones_like(kl_tensor) * reward_val - kl_coef * kl_tensor
        for kl_tensor, reward_val in zip(kl, rewards, strict=False)
    ]

    return unwhitened_advantages


def get_advantages_and_returns(
    total_len: int,
    response_len: int,
    values: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float,
    lambd: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function that computes advantages and returns from rewards and values.
    Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
    Note that rewards may include a KL divergence loss term.

    Advantages looks like this:
    Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
            - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Returns looks like this:
    Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Input:
    - values: Tensor of shape (response_size,)
    - rewards: Tensor of shape (response_size,)

    Output:
    - advantages: Tensor of shape (response_size,)
    - returns: Tensor of shape (response_size,)
    """
    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()
    if cp_size > 1:
        from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

        full_rewards = all_gather_with_cp(rewards, total_len, response_len)
        full_values = all_gather_with_cp(values, total_len, response_len)
    else:
        full_rewards = rewards
        full_values = values

    lastgaelam = 0
    advantages_reversed = []

    for t in reversed(range(response_len)):
        nextvalues = full_values[t + 1] if t < response_len - 1 else 0.0
        delta = full_rewards[t] + gamma * nextvalues - full_values[t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        advantages_reversed.append(lastgaelam)
    full_advantages = torch.tensor(advantages_reversed[::-1], dtype=full_values.dtype, device=full_values.device)
    full_returns = full_advantages + full_values

    if cp_size > 1:
        from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

        advantages = slice_log_prob_with_cp(full_advantages, total_len, response_len)
        returns = slice_log_prob_with_cp(full_returns, total_len, response_len)
    else:
        advantages = full_advantages
        returns = full_returns

    return advantages.detach(), returns


def get_advantages_and_returns_batch(
    total_lengths,
    response_lengths,
    values_list,
    rewards_list,
    gamma,
    lambd,
    chunked: bool = True,
):
    """
    Batched GAE with CP support.
    Input:
        total_lengths:     list[int], each sample's total_len
        response_lengths:  list[int], each sample's response_len
        values_list:       list[Tensor], each shape = [resp_len_i]
        rewards_list:      list[Tensor], same shape
    Output:
        advantages_list:   list[Tensor], each shape = [resp_len_i]
        returns_list:      list[Tensor], same shape
    """

    from megatron.core import mpu

    with torch.no_grad():
        B = len(response_lengths)
        assert B == len(values_list)
        assert B == len(rewards_list)

        cp_size = mpu.get_context_parallel_world_size()
        device = values_list[0].device
        dtype = values_list[0].dtype

        if cp_size > 1:
            from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

            full_values_list = []
            full_rewards_list = []

            for total_len, resp_len, v, r in zip(
                total_lengths, response_lengths, values_list, rewards_list, strict=False
            ):
                full_v = all_gather_with_cp(v, total_len, resp_len)
                full_r = all_gather_with_cp(r, total_len, resp_len)
                full_values_list.append(full_v)
                full_rewards_list.append(full_r)

            # full_values_list[i].shape = [total_len_i]
        else:
            full_values_list = values_list
            full_rewards_list = rewards_list

        # pad to max_len for batched GAE
        max_len = max(response_lengths)

        full_values = torch.zeros(B, max_len, device=device, dtype=dtype)
        full_rewards = torch.zeros(B, max_len, device=device, dtype=dtype)

        for i in range(B):
            L = response_lengths[i]
            full_values[i, :L] = full_values_list[i][:L]
            full_rewards[i, :L] = full_rewards_list[i][:L]

        if not chunked:
            full_advantages, full_returns = vanilla_gae(
                rewards=full_rewards,
                values=full_values,
                gamma=gamma,
                lambd=lambd,
            )
        else:
            full_advantages, full_returns = chunked_gae(
                rewards=full_rewards,
                values=full_values,
                gamma=gamma,
                lambd=lambd,
            )

        advantages_list = []
        returns_list = []

        if cp_size > 1:
            from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

            for total_len, resp_len, adv_row, ret_row in zip(
                total_lengths,
                response_lengths,
                full_advantages,
                full_returns,
                strict=False,
            ):
                adv_full = adv_row  # shape = [resp_len_i padded to max_len]
                ret_full = ret_row

                adv_sliced = slice_log_prob_with_cp(adv_full[:resp_len], total_len, resp_len)
                ret_sliced = slice_log_prob_with_cp(ret_full[:resp_len], total_len, resp_len)

                advantages_list.append(adv_sliced)
                returns_list.append(ret_sliced)

        else:
            for i in range(B):
                L = response_lengths[i]
                advantages_list.append(full_advantages[i, :L])
                returns_list.append(full_returns[i, :L])

    return advantages_list, returns_list


def vanilla_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd: float,
):
    B, T = rewards.shape
    device = rewards.device
    dtype = rewards.dtype

    lastgaelam = torch.zeros(B, device=device, dtype=dtype)
    adv_rev = []

    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t < T - 1 else 0.0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        adv_rev.append(lastgaelam)

    full_advantages = torch.stack(adv_rev[::-1], dim=1)  # [B, max_len]
    full_returns = full_advantages + values  # [B, max_len]
    return full_advantages, full_returns


def chunked_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd: float,
    chunk_size: int = 128,
):
    """
    Compute Generalized Advantage Estimation (GAE) using a FlashLinearAttention-
    inspired algorithm: parallel prefix scan within chunks and recurrent state
    propagation across chunks.

    This reduces the sequential dependency length from O(T) to O(T / chunk_size),
    while keeping chunk computations fully parallelizable (O(C^2) per chunk).

    Args:
        rewards (Tensor): [B, T] reward sequence.
        values (Tensor):  [B, T] value predictions. The next-value of the final
                          step is assumed to be zero (standard PPO convention).
        gamma (float): discount factor.
        lam (float): GAE lambda.
        chunk_size (int): sequence chunk length for parallel scan.

    Returns:
        advantages (Tensor): [B, T] computed advantages.
        returns (Tensor):    [B, T] advantages + values.
    """

    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------
    assert rewards.ndim == 2 and values.ndim == 2
    B, T = rewards.shape
    assert values.shape == (B, T)

    device = rewards.device
    dtype = rewards.dtype

    # -------------------------------------------------------------------------
    # Build δ_t = r_t + γ * V_{t+1} - V_t   with V_{T} = 0
    # -------------------------------------------------------------------------
    next_values = torch.cat(
        [values[:, 1:], torch.zeros(B, 1, device=device, dtype=dtype)],
        dim=1,
    )
    deltas = rewards + gamma * next_values - values

    # Reformulate backward GAE as a forward scan on the reversed sequence:
    #   S[i] = Δ[i] + w * S[i - 1],   w = γλ
    w = gamma * lambd
    deltas_rev = torch.flip(deltas, dims=[1])  # [B, T]

    # -------------------------------------------------------------------------
    # Pad to a multiple of chunk_size
    # -------------------------------------------------------------------------
    if T % chunk_size != 0:
        pad = chunk_size - (T % chunk_size)
        deltas_rev = F.pad(deltas_rev, (0, pad))
    else:
        pad = 0

    B, T_pad = deltas_rev.shape
    n_chunks = T_pad // chunk_size

    deltas_chunks = deltas_rev.view(B, n_chunks, chunk_size)

    # -------------------------------------------------------------------------
    # Construct the intra-chunk parallel scan kernel M
    #
    # For a chunk Δ[0..C-1], we want:
    #   S_local[t] = sum_{k=0..t} w^(t-k) * Δ[k]
    #
    # This is implemented as:
    #   S_local = Δ @ M
    #
    # where:
    #   M[i, j] = w^(j - i)    if j >= i
    #             0            otherwise
    # -------------------------------------------------------------------------
    idx = torch.arange(chunk_size, device=device)
    row = idx[:, None]
    col = idx[None, :]
    diff = col - row

    M = torch.zeros(chunk_size, chunk_size, device=device, dtype=dtype)
    mask = diff >= 0

    if w == 0.0:
        M[mask & (diff == 0)] = 1.0
    else:
        M[mask] = w ** diff[mask].to(dtype)

    # pow_vec[t] = w^(t+1), used to inject the recurrent state s_prev
    if w == 0.0:
        pow_vec = torch.zeros(chunk_size, device=device, dtype=dtype)
    else:
        pow_vec = w ** torch.arange(1, chunk_size + 1, device=device, dtype=dtype)

    # -------------------------------------------------------------------------
    # Parallel compute local chunk results (assuming initial state = 0)
    # -------------------------------------------------------------------------
    deltas_flat = deltas_chunks.reshape(B * n_chunks, chunk_size)
    S_local_flat = deltas_flat @ M
    S_local_chunks = S_local_flat.view(B, n_chunks, chunk_size)

    # Effective length of each chunk (the last chunk may be padded)
    lengths = [chunk_size] * n_chunks
    if pad > 0:
        lengths[-1] = chunk_size - pad

    # -------------------------------------------------------------------------
    # Recurrent propagation between chunks
    #
    # Each chunk contributes:
    #   S_global[t] = S_local[t] + w^(t+1) * s_prev
    #
    # And updates:
    #   s_prev = S_global[last_t]
    # -------------------------------------------------------------------------
    S_rev = deltas_rev.new_zeros(B, T_pad)
    s_prev = torch.zeros(B, device=device, dtype=dtype)

    for c in range(n_chunks):
        Lc = lengths[c]
        start = c * chunk_size
        end = start + Lc

        S_local = S_local_chunks[:, c, :Lc]
        S_global = S_local + s_prev.unsqueeze(1) * pow_vec[:Lc]

        S_rev[:, start:end] = S_global
        s_prev = S_global[:, -1]  # state for next chunk

    # Remove padding and flip back to original time order
    if pad > 0:
        S_rev = S_rev[:, :T]

    advantages = torch.flip(S_rev, dims=[1])
    returns = advantages + values

    return advantages, returns


def calculate_log_probs_and_entropy(
    logits,
    tokens,
    tp_group,
    with_entropy: bool = False,
    chunk_size: int = -1,
    with_dppo_directional_moment: bool = False,
):
    if with_dppo_directional_moment and not with_entropy:
        raise ValueError("DPPO entropy directional moments require with_entropy=True.")
    logits = logits.contiguous()
    entropy = None
    dppo_directional_moment = None
    if logits.size(0) != 0:
        if chunk_size > 0:
            num_chunks = (logits.size(0) - 1) // chunk_size + 1
            logits_chunks = logits.chunk(num_chunks, dim=0)
            tokens_chunks = tokens.chunk(num_chunks, dim=0)

            if with_entropy:
                entropys = []
                dppo_directional_moments = []
                for logits_chunk in logits_chunks:
                    entropy_input = logits_chunk.clone()
                    if with_dppo_directional_moment:
                        entropy_chunk, moment_chunk = compute_entropy_and_dppo_directional_moment_from_logits(
                            entropy_input, tp_group
                        )
                        entropys.append(entropy_chunk)
                        dppo_directional_moments.append(moment_chunk)
                    else:
                        entropys.append(compute_entropy_from_logits(entropy_input, tp_group))
                entropy = torch.cat(entropys, dim=0)
                if with_dppo_directional_moment:
                    dppo_directional_moment = torch.cat(dppo_directional_moments, dim=0)

            log_probs = []
            for tokens_chunk, logits_chunk in zip(tokens_chunks, logits_chunks, strict=True):
                log_prob = compute_log_probs(logits_chunk.clone(), tokens_chunk, tp_group)
                log_probs.append(log_prob)
            log_prob = torch.cat(log_probs, dim=0)
        else:
            if with_entropy:
                entropy_input = logits.clone()
                if with_dppo_directional_moment:
                    entropy, dppo_directional_moment = compute_entropy_and_dppo_directional_moment_from_logits(
                        entropy_input, tp_group
                    )
                else:
                    entropy = compute_entropy_from_logits(entropy_input, tp_group)

            log_prob = compute_log_probs(logits.clone(), tokens, tp_group)
    else:
        log_prob = logits.new_zeros((0,))
        if with_entropy:
            entropy = logits.new_zeros((0,))
        if with_dppo_directional_moment:
            dppo_directional_moment = logits.new_zeros((0,))

    if with_dppo_directional_moment:
        return log_prob, entropy, dppo_directional_moment
    return log_prob, entropy
