# Adapt from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/models/utils.py
# and https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ppo_utils/experience_maker.py

import math
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


class PolicyLossOutput(dict):
    """Policy-loss metrics with tuple-unpacking compatibility.

    Official slime callers unpack ``(pg_losses, pg_clipfrac)`` while the
    extended policy modes consume named clipping metrics. Keep both APIs.
    """

    def __iter__(self):
        yield self["pg_losses"]
        yield self["pg_clipfrac"]


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
def _compute_policy_loss_tensors(
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
    upper_clipfrac = clipfrac * (ratio > 1 + eps_clip_high).float()
    lower_clipfrac = clipfrac * (ratio < 1 - eps_clip).float()

    if eps_clip_c is not None:
        assert (
            eps_clip_c > 1.0
        ), f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {eps_clip_c}."
        pg_losses3 = -eps_clip_c * advantages
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    else:
        pg_losses = clip_pg_losses1

    return pg_losses, clipfrac, upper_clipfrac, lower_clipfrac


def compute_policy_loss(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
) -> PolicyLossOutput:
    pg_losses, clipfrac, upper_clipfrac, lower_clipfrac = _compute_policy_loss_tensors(
        ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c
    )
    return PolicyLossOutput(
        pg_losses=pg_losses,
        pg_clipfrac=clipfrac,
        pg_upper_clipfrac=upper_clipfrac,
        pg_lower_clipfrac=lower_clipfrac,
    )


def _compute_policy_loss_eager(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
) -> PolicyLossOutput:
    """Eager equivalent exposed through ``compute_policy_loss.__wrapped__`` for CPU tests."""
    pg_losses, clipfrac, upper_clipfrac, lower_clipfrac = _compute_policy_loss_tensors.__wrapped__(
        ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c
    )
    return PolicyLossOutput(
        pg_losses=pg_losses,
        pg_clipfrac=clipfrac,
        pg_upper_clipfrac=upper_clipfrac,
        pg_lower_clipfrac=lower_clipfrac,
    )


compute_policy_loss.__wrapped__ = _compute_policy_loss_eager


def compute_policy_loss_output(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
) -> dict[str, torch.Tensor]:
    """Return ordinary PPO loss plus the richer metric mapping used by newer modes.

    ``compute_policy_loss`` retains slime's established two-tensor return
    protocol; this adapter supplies the mapping protocol used by the newer
    policy-loss implementations.
    """
    return compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c)


def compute_up_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
):
    """Compute UP asymmetric policy loss.

    Implements UP: Unbounded Positive Asymmetric Optimization for Breaking
    the Exploration-Stability Dilemma (arXiv:2607.06987). Positive advantages
    use an unclipped REINFORCE-style log-prob objective. Non-positive
    advantages keep the standard PPO/DAPO clipped objective as the
    trust-region safeguard. Reported epsilon settings: UP-DAPO uses
    eps_clip=0.2 and no positive upper clip; UP-GRPO uses eps_clip=0.2 and
    no positive upper clip; UP-GSPO uses eps_clip=3e-4 and no positive upper
    clip.
    """
    ppo_output = compute_policy_loss_output(old_log_probs - log_probs, advantages, eps_clip, eps_clip_high, eps_clip_c)
    positive_advantage_mask = advantages > 0
    positive_pg_losses = -advantages * log_probs
    pg_losses = torch.where(positive_advantage_mask, positive_pg_losses, ppo_output["pg_losses"])

    return {
        "pg_losses": pg_losses,
        "pg_clipfrac": torch.where(positive_advantage_mask, torch.zeros_like(advantages), ppo_output["pg_clipfrac"]),
        "pg_upper_clipfrac": torch.where(
            positive_advantage_mask, torch.zeros_like(advantages), ppo_output["pg_upper_clipfrac"]
        ),
        "pg_lower_clipfrac": torch.where(
            positive_advantage_mask, torch.zeros_like(advantages), ppo_output["pg_lower_clipfrac"]
        ),
    }


def compute_aspo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
):
    """Compute ASPO loss with asymmetric positive-token IS weights.

    Implements ASPO from When Importance Sampling Misallocates Credit:
    Asymmetric Ratios for Outcome-Supervised RL (arXiv:2510.06062). Negative
    advantages keep the standard PPO/GRPO active-region gradient. Positive
    advantages use reciprocal ratio weights, hard-mask high-ratio over-updates,
    and optionally soft dual-clip the reciprocal weight. Reported experiments
    use eps_clip=0.2 for GRPO/ASPO clipping; explicit KL uses k3 with
    coefficient 0.001. The paper ablates dual clipping but does not publish a
    separate numeric reciprocal dual-clip threshold.
    """
    ratio = torch.exp(torch.clamp(log_probs - old_log_probs, min=-20.0, max=20.0))
    positive_advantage_mask = advantages > 0

    negative_output = compute_policy_loss_output(old_log_probs - log_probs, advantages, eps_clip, eps_clip_high, None)

    invalid_positive_mask = positive_advantage_mask & (ratio > 1 + eps_clip_high)
    positive_valid_mask = 1.0 - invalid_positive_mask.detach().float()
    reciprocal_ratio = (1.0 / ratio).detach()
    if eps_clip_c is not None:
        assert (
            eps_clip_c > 1.0
        ), f"The upper bound of ASPO reciprocal dual-clip should be greater than 1.0, but get the value: {eps_clip_c}."
        reciprocal_ratio = reciprocal_ratio.clamp(max=eps_clip_c)
    positive_pg_losses = -advantages * reciprocal_ratio * positive_valid_mask * log_probs

    pg_losses = torch.where(positive_advantage_mask, positive_pg_losses, negative_output["pg_losses"])
    upper_clipfrac = torch.where(
        positive_advantage_mask, invalid_positive_mask.float(), negative_output["pg_upper_clipfrac"]
    )
    lower_clipfrac = torch.where(
        positive_advantage_mask, torch.zeros_like(advantages), negative_output["pg_lower_clipfrac"]
    )
    clipfrac = torch.maximum(upper_clipfrac, lower_clipfrac)

    return {
        "pg_losses": pg_losses,
        "pg_clipfrac": clipfrac,
        "pg_upper_clipfrac": upper_clipfrac,
        "pg_lower_clipfrac": lower_clipfrac,
    }


def compute_ripo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    ripo_delta: float,
    ripo_delta_high: float,
    ripo_ratio_min: float | None = 0.5,
    ripo_ratio_max: float | None = 10.0,
):
    """Compute RIPO/RIC policy loss.

    Implements Riemannian Isometric Policy Optimization from Beyond Euclidean
    Clipping: Overcoming Exploration Collapse in LLM RL via Riemannian
    Isometric Policy Optimization (arXiv:2607.10169). RIPO replaces fixed PPO
    clipping with token-wise Riemannian Isometric Clip boundaries
    eps_i,t=sqrt(delta/pi_old(token)). Reported experiments use delta=0.05
    by default and outer ratio clipping bounds [0.5, 10].
    """
    assert ripo_delta > 0.0, f"ripo_delta must be positive, got {ripo_delta}."
    assert ripo_delta_high > 0.0, f"ripo_delta_high must be positive, got {ripo_delta_high}."

    log_ratio = torch.clamp(log_probs - old_log_probs, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    old_prob = old_log_probs.float().exp().detach().clamp_min(1e-12).to(ratio.dtype)

    eps_low = torch.sqrt(ratio.new_tensor(ripo_delta) / old_prob)
    eps_high = torch.sqrt(ratio.new_tensor(ripo_delta_high) / old_prob)
    clip_lower = 1.0 - eps_low
    clip_upper = 1.0 + eps_high

    if ripo_ratio_min is not None:
        clip_lower = torch.clamp(clip_lower, min=ripo_ratio_min)
    if ripo_ratio_max is not None:
        clip_upper = torch.clamp(clip_upper, max=ripo_ratio_max)

    clipped_ratio = torch.minimum(torch.maximum(ratio, clip_lower), clip_upper)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * clipped_ratio
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    clipfrac = (pg_losses2 > pg_losses1).float()
    upper_clipfrac = clipfrac * (ratio > clip_upper).float()
    lower_clipfrac = clipfrac * (ratio < clip_lower).float()

    return {
        "pg_losses": pg_losses,
        "pg_clipfrac": clipfrac,
        "pg_upper_clipfrac": upper_clipfrac,
        "pg_lower_clipfrac": lower_clipfrac,
        "ripo_eps_low": eps_low,
        "ripo_eps_high": eps_high,
        "ripo_clip_lower": clip_lower,
        "ripo_clip_upper": clip_upper,
    }


def compute_cppo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    delta: float,
    prefix_delta: float,
    weight_floor: float,
    eps_clip_c: float | None = None,
):
    """Compute CPPO policy loss in Binary-TV mode for one complete response.

    Implements Cumulative Prefix-divergence Policy Optimization from Beyond Uniform
    Token-Level Trust Region in LLM Reinforcement Learning
    (arXiv:2606.10968). The sampled-token Binary-TV divergence is weighted by
    response position and admitted only when both its token threshold and the
    preceding prefix budget allow it. This implementation follows the official
    UniRL cppo.py Binary-TV mode. The prefix floor is calibrated per sequence
    as clamp(P90(D), prefix_delta, 2*prefix_delta), matching the official UniRL
    implementation. Reported experiments use weight_floor=0.8, delta=0.15
    (0.2 for the 30B MoE), and prefix_delta=0.02 for Base models.
    """
    if log_probs.ndim != 1 or old_log_probs.ndim != 1 or advantages.ndim != 1:
        raise ValueError("CPPO expects one-dimensional tensors for one complete response.")
    if not (log_probs.shape == old_log_probs.shape == advantages.shape):
        raise ValueError(
            "CPPO tensor shapes must match, got "
            f"log_probs={log_probs.shape}, old_log_probs={old_log_probs.shape}, advantages={advantages.shape}."
        )
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError(f"CPPO delta must be finite and positive, got {delta}.")
    if not math.isfinite(prefix_delta) or prefix_delta <= 0.0:
        raise ValueError(f"CPPO prefix_delta must be finite and positive, got {prefix_delta}.")
    if not math.isfinite(weight_floor) or not 0.0 < weight_floor <= 1.0:
        raise ValueError(f"CPPO weight_floor must be finite and in (0, 1], got {weight_floor}.")

    response_length = log_probs.numel()
    old_prob = old_log_probs.float().exp().detach()
    divergence = (log_probs.float().exp().detach() - old_prob).abs()

    if response_length <= 1:
        position_weight = torch.ones_like(log_probs)
    else:
        positions = torch.arange(response_length, dtype=divergence.dtype, device=log_probs.device)
        position_weight = 1.0 - (1.0 - weight_floor) * positions / (response_length - 1)

    weighted_divergence = position_weight * divergence
    prefix_divergence = torch.cat([weighted_divergence.new_zeros(1), weighted_divergence.cumsum(dim=0)[:-1]], dim=0)
    prefix_weight = torch.cat([position_weight.new_zeros(1), position_weight.cumsum(dim=0)[:-1]], dim=0)
    sequence_prefix_delta = torch.quantile(divergence, 0.9).clamp(prefix_delta, 2.0 * prefix_delta)
    effective_threshold = torch.minimum(
        weighted_divergence.new_full(weighted_divergence.shape, delta),
        delta + sequence_prefix_delta * prefix_weight - prefix_divergence,
    )

    ratio_clip_c = 20.0 if eps_clip_c is None else eps_clip_c
    ratio = torch.exp(torch.clamp(log_probs - old_log_probs, min=-20.0, max=20.0))
    ratio = torch.clamp(ratio, max=ratio_clip_c).detach()

    outward_update = advantages * (ratio.detach() - 1.0) > 0.0
    invalid_mask = outward_update & (weighted_divergence > effective_threshold)
    valid_mask = (~invalid_mask).to(log_probs.dtype)

    pg_losses = -advantages * ratio * valid_mask * log_probs

    positive_advantage_mask = advantages > 0
    token_threshold_violation = outward_update & (weighted_divergence > delta)
    prefix_threshold_violation = invalid_mask & ~token_threshold_violation
    return {
        "pg_losses": pg_losses,
        "pg_clipfrac": invalid_mask.float(),
        "pg_upper_clipfrac": (positive_advantage_mask & invalid_mask).float(),
        "pg_lower_clipfrac": (~positive_advantage_mask & invalid_mask).float(),
        "cppo_token_clipfrac": token_threshold_violation.float(),
        "cppo_prefix_clipfrac": prefix_threshold_violation.float(),
        "cppo_divergence": divergence,
        "cppo_weighted_divergence": weighted_divergence,
        "cppo_effective_threshold": effective_threshold,
        "cppo_prefix_delta": torch.full_like(divergence, sequence_prefix_delta),
    }


def compute_cispo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
):
    """Compute CISPO policy loss with a detached clipped importance ratio."""
    ratio = torch.exp(torch.clamp(log_probs - old_log_probs, min=-20.0, max=20.0))
    clipped_ratio = torch.clamp(ratio, min=1 - eps_clip, max=1 + eps_clip_high)
    clipped_ratio_sg = clipped_ratio.detach()

    pg_losses = -clipped_ratio_sg * advantages * log_probs
    upper_clipfrac = (ratio > 1 + eps_clip_high).float()
    lower_clipfrac = (ratio < 1 - eps_clip).float()
    clipfrac = torch.maximum(upper_clipfrac, lower_clipfrac)
    return {
        "pg_losses": pg_losses,
        "pg_clipfrac": clipfrac,
        "pg_upper_clipfrac": upper_clipfrac,
        "pg_lower_clipfrac": lower_clipfrac,
    }


def compute_dppo_binary_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    loss_mode: str,
    eps_clip_c: float | None = None,
):
    """Compute DPPO Binary-TV/KL policy loss.

    Implements Divergence Proximal Policy Optimization from Rethinking the
    Trust Region in LLM Reinforcement Learning (arXiv:2602.04879). Binary
    variants split the vocabulary into the sampled token and all other tokens,
    then mask updates that cross the divergence threshold in the
    advantage-driven direction. Reported scaling experiments use delta=0.2 for
    Binary-TV (0.15 for the MoE Base with LoRA experiment) and delta=0.05 for
    Binary-KL.
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


def compute_drpo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
):
    """Compute DRPO smooth Binary-TV regularized policy loss.

    Implements Divergence Regularized Policy Optimization from Rethinking the
    Divergence Regularization in LLM RL (arXiv:2606.09821). DRPO keeps DPPO's
    sampled-token Binary-TV trust region but replaces its hard update mask with
    an advantage-weighted quadratic regularizer. Reported experiments use a
    symmetric regularization threshold delta=12.5 and ablate delta=2.5.
    """
    ratio = torch.exp(torch.clamp(log_probs - old_log_probs, min=-20.0, max=20.0))
    old_prob = old_log_probs.float().exp().detach()

    positive_advantage_mask = advantages > 0
    eps = torch.where(
        positive_advantage_mask,
        torch.full_like(advantages, eps_clip_high),
        torch.full_like(advantages, eps_clip),
    )
    eps = torch.clamp(eps, min=1e-8)

    quadratic_penalty = advantages.abs() * old_prob * (ratio - 1.0).pow(2) / (2.0 * eps)
    pg_losses = -advantages * ratio + quadratic_penalty

    prob = log_probs.float().exp()
    invalid_positive_mask = (prob - old_prob) > eps_clip_high
    invalid_negative_mask = (prob - old_prob) < -eps_clip
    invalid_mask = torch.where(positive_advantage_mask, invalid_positive_mask, invalid_negative_mask)

    upper_clipfrac = (positive_advantage_mask & invalid_positive_mask).float()
    lower_clipfrac = (~positive_advantage_mask & invalid_negative_mask).float()
    return {
        "pg_losses": pg_losses,
        "pg_clipfrac": invalid_mask.float(),
        "pg_upper_clipfrac": upper_clipfrac,
        "pg_lower_clipfrac": lower_clipfrac,
    }


@torch.compile(dynamic=True)
def compute_cispo_loss(
    ppo_kl: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
):
    """CISPO loss from MiniMax-M1 (https://arxiv.org/abs/2506.13585, Eq. 4-5):
    ``-sg(clip(ratio, 1 - eps_clip, 1 + eps_clip_high)) * advantages * log_probs``.

    Unlike PPO, the IS ratio is clipped under stop-gradient and the gradient flows
    through ``log_probs``, so clipped tokens still contribute gradient. The bounds
    reuse the delta-from-1 convention of ``compute_policy_loss``; canonical CISPO
    disables the lower bound (``eps_clip >= 1.0``).
    """
    ratio = (-ppo_kl).exp()
    ratio_truncated = torch.clamp(ratio, min=1.0 - eps_clip, max=1.0 + eps_clip_high)
    pg_losses = -ratio_truncated.detach() * advantages * log_probs
    clipfrac = (ratio_truncated != ratio).float()
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


def _maybe_all_reduce(tensor: torch.Tensor, op: dist.ReduceOp, process_group) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=op, group=process_group)


def _get_vocab_parallel_rank_size(process_group) -> tuple[int, int]:
    if process_group is not None and hasattr(process_group, "rank") and hasattr(process_group, "size"):
        return process_group.rank(), process_group.size()
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group=process_group), dist.get_world_size(group=process_group)
    return 0, 1


class _VocabParallelLogProbEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        log_prob_keep_mask: torch.Tensor | None,
        process_group,
        with_entropy: bool,
        with_entropy_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with_entropy_grad = with_entropy and with_entropy_grad
        vocab_parallel_logits = vocab_parallel_logits.float()
        seq_len, vocab_parallel_size = vocab_parallel_logits.shape
        rank, _world_size = _get_vocab_parallel_rank_size(process_group)
        vocab_start_index = rank * vocab_parallel_size
        vocab_end_index = vocab_start_index + vocab_parallel_size

        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target_1d = (target - vocab_start_index).clone()
        masked_target_1d[target_mask] = 0
        arange_1d = torch.arange(seq_len, device=vocab_parallel_logits.device)

        def vocab_parallel_softmax(
            logits: torch.Tensor,
            inplace: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            logits_max = logits.max(dim=-1, keepdim=True).values
            _maybe_all_reduce(logits_max, dist.ReduceOp.MAX, process_group)
            # Subtract the max for numerical stability. When ``inplace`` is set, the
            # caller passed a scratch buffer it owns, so overwrite it instead of
            # allocating another [seq_len, vocab] tensor.
            normalized_logits = logits.sub_(logits_max) if inplace else logits - logits_max
            # The normalized logit at the target position is the log-prob numerator;
            # gather it (a small copy) before the in-place ``exp_`` destroys it.
            predicted_logits = normalized_logits.view(-1, vocab_parallel_size)[arange_1d, masked_target_1d]
            # Reuse the ``normalized_logits`` storage for exp and softmax so the whole
            # softmax costs a single [seq_len, vocab] buffer instead of three.
            exp_logits = normalized_logits.exp_()
            sum_exp_logits = exp_logits.sum(dim=-1, keepdim=True)
            _maybe_all_reduce(sum_exp_logits, dist.ReduceOp.SUM, process_group)
            softmax = exp_logits.div_(sum_exp_logits)
            return predicted_logits, sum_exp_logits, softmax, logits_max

        entropy = vocab_parallel_logits.new_zeros((0,))
        entropy_softmax = vocab_parallel_logits.new_empty((0,))
        sum_softmax_times_logits = vocab_parallel_logits.new_empty((0,))

        def sum_softmax_logits(softmax: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
            if softmax.is_cuda:
                # Avoid materializing the full [seq_len, vocab] product buffer.
                return torch.einsum("ij,ij->i", softmax, logits).unsqueeze(-1)
            return (softmax * logits).sum(dim=-1, keepdim=True)

        if log_prob_keep_mask is None:
            predicted_logits, log_prob_sum_exp_logits, log_prob_softmax, log_prob_logits_max = vocab_parallel_softmax(
                vocab_parallel_logits
            )
            if with_entropy:
                entropy_softmax = log_prob_softmax
                sum_softmax_times_logits = sum_softmax_logits(entropy_softmax, vocab_parallel_logits)
                _maybe_all_reduce(sum_softmax_times_logits, dist.ReduceOp.SUM, process_group)
                entropy = log_prob_logits_max + log_prob_sum_exp_logits.log() - sum_softmax_times_logits
                entropy = entropy.squeeze(dim=-1)
        else:
            if with_entropy:
                _entropy_predicted_logits, entropy_sum_exp_logits, entropy_softmax, entropy_logits_max = (
                    vocab_parallel_softmax(vocab_parallel_logits)
                )
                sum_softmax_times_logits = sum_softmax_logits(entropy_softmax, vocab_parallel_logits)
                _maybe_all_reduce(sum_softmax_times_logits, dist.ReduceOp.SUM, process_group)
                entropy = entropy_logits_max + entropy_sum_exp_logits.log() - sum_softmax_times_logits
                entropy = entropy.squeeze(dim=-1)

            local_target_rows = torch.nonzero(~target_mask, as_tuple=False).squeeze(-1)
            log_prob_logits = vocab_parallel_logits.masked_fill(~log_prob_keep_mask, float("-inf"))
            if local_target_rows.numel() > 0:
                log_prob_logits[local_target_rows, masked_target_1d[local_target_rows]] = vocab_parallel_logits[
                    local_target_rows, masked_target_1d[local_target_rows]
                ]
            # ``log_prob_logits`` is an owned scratch buffer here, so let the softmax
            # consume it in place rather than allocating another copy.
            predicted_logits, log_prob_sum_exp_logits, log_prob_softmax, _log_prob_logits_max = vocab_parallel_softmax(
                log_prob_logits, inplace=True
            )

        predicted_logits = predicted_logits.masked_fill_(target_mask, 0.0).unsqueeze(-1)
        _maybe_all_reduce(predicted_logits, dist.ReduceOp.SUM, process_group)
        log_prob = predicted_logits - log_prob_sum_exp_logits.log()

        if not with_entropy_grad:
            ctx.mark_non_differentiable(entropy)

        ctx.with_entropy_grad = with_entropy_grad
        # Metric-only entropy still returns values, but does not need the
        # full-vocab entropy tensors kept alive for backward.
        saved_entropy_softmax = entropy_softmax if with_entropy_grad else vocab_parallel_logits.new_empty((0,))
        saved_sum_softmax_times_logits = (
            sum_softmax_times_logits if with_entropy_grad else vocab_parallel_logits.new_empty((0,))
        )
        saved_logits = vocab_parallel_logits if with_entropy_grad else vocab_parallel_logits.new_empty((0,))
        ctx.save_for_backward(
            log_prob_softmax,
            target_mask,
            masked_target_1d,
            saved_entropy_softmax,
            saved_sum_softmax_times_logits,
            saved_logits,
        )
        return log_prob, entropy

    @staticmethod
    def backward(
        ctx, grad_log_prob: torch.Tensor | None, grad_entropy: torch.Tensor | None
    ) -> tuple[torch.Tensor, None, None, None, None, None]:
        (
            log_prob_softmax,
            target_mask,
            masked_target_1d,
            entropy_softmax,
            sum_softmax_times_logits,
            vocab_parallel_logits,
        ) = ctx.saved_tensors

        if grad_log_prob is None:
            raise RuntimeError(
                "_VocabParallelLogProbEntropy expected a materialized grad_log_prob. "
                "Do not call ctx.set_materialize_grads(False)."
            )

        grad_entropy_input = None
        if ctx.with_entropy_grad and grad_entropy is not None and grad_entropy.numel() > 0:
            # In the unmasked path, entropy_softmax aliases log_prob_softmax.
            # Build entropy grad before mutating log_prob_softmax below.
            grad_entropy_input = sum_softmax_times_logits - vocab_parallel_logits
            grad_entropy_input.mul_(entropy_softmax)
            grad_entropy_input.mul_(grad_entropy.reshape(-1, 1))

        vocab_parallel_size = log_prob_softmax.size(-1)
        grad_input = log_prob_softmax.neg_()
        grad_2d = grad_input.view(-1, vocab_parallel_size)
        arange_1d = torch.arange(grad_2d.size(0), device=grad_2d.device)
        target_update = (~target_mask).to(dtype=grad_2d.dtype)
        grad_2d[arange_1d, masked_target_1d] += target_update
        grad_input.mul_(grad_log_prob.reshape(-1, 1))

        if grad_entropy_input is not None:
            grad_input.add_(grad_entropy_input)

        return grad_input, None, None, None, None, None


def _calculate_log_probs_and_entropy_chunk(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    tp_group,
    *,
    with_entropy: bool,
    with_entropy_grad: bool = True,
    log_prob_keep_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    log_prob, entropy = _VocabParallelLogProbEntropy.apply(
        logits,
        tokens,
        log_prob_keep_mask,
        tp_group,
        with_entropy,
        with_entropy_grad,
    )
    if not with_entropy:
        entropy = None
    return log_prob, entropy


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

    token_level_rewards = []
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
        rewards_for_seq = -kl_coef * masked_kl
        last_idx = full_mask.nonzero(as_tuple=True)[0][-1]
        rewards_for_seq[last_idx] += rewards[i]
        token_level_rewards.append(rewards_for_seq)

    if not token_level_rewards:
        return []

    max_len = max(rewards_for_seq.size(0) for rewards_for_seq in token_level_rewards)
    padded_rewards = token_level_rewards[0].new_zeros(len(token_level_rewards), max_len)
    for i, rewards_for_seq in enumerate(token_level_rewards):
        padded_rewards[i, : rewards_for_seq.size(0)] = rewards_for_seq

    padded_returns = chunked_discounted_returns(padded_rewards, gamma)

    final_returns_chunks = []
    for i, returns_for_seq in enumerate(padded_returns):
        returns_for_seq = returns_for_seq[: token_level_rewards[i].size(0)]
        if cp_size > 1:
            from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

            total_len, response_len = total_lengths[i], response_lengths[i]
            local_returns_chunk = slice_log_prob_with_cp(returns_for_seq, total_len, response_len)
        else:
            local_returns_chunk = returns_for_seq

        final_returns_chunks.append(local_returns_chunk)

    return final_returns_chunks


def get_reinforce_plus_plus_baseline_advantages(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
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


def chunked_discounted_returns(
    rewards: torch.Tensor,
    discount: float,
    chunk_size: int = 128,
) -> torch.Tensor:
    """
    Compute discounted returns using a parallel scan within fixed-size chunks.

    This reduces the sequential dependency length from O(T) to O(T / chunk_size),
    while keeping chunk computations fully parallelizable (O(C^2) per chunk).

    Args:
        rewards (Tensor): [B, T] reward sequence.
        discount (float): Discount factor applied at each step.
        chunk_size (int): sequence chunk length for parallel scan.

    Returns:
        Tensor: [B, T] discounted returns.
    """
    assert rewards.ndim == 2
    B, T = rewards.shape

    device = rewards.device
    dtype = rewards.dtype

    # Reformulate the backward recurrence as a forward scan on the reversed
    # sequence: S[i] = rewards[i] + discount * S[i - 1].
    rewards_rev = torch.flip(rewards, dims=[1])

    if T % chunk_size != 0:
        pad = chunk_size - (T % chunk_size)
        rewards_rev = F.pad(rewards_rev, (0, pad))
    else:
        pad = 0

    B, T_pad = rewards_rev.shape
    n_chunks = T_pad // chunk_size
    rewards_chunks = rewards_rev.view(B, n_chunks, chunk_size)

    idx = torch.arange(chunk_size, device=device)
    row = idx[:, None]
    col = idx[None, :]
    diff = col - row

    M = torch.zeros(chunk_size, chunk_size, device=device, dtype=dtype)
    mask = diff >= 0

    if discount == 0.0:
        M[mask & (diff == 0)] = 1.0
    else:
        M[mask] = discount ** diff[mask].to(dtype)

    if discount == 0.0:
        pow_vec = torch.zeros(chunk_size, device=device, dtype=dtype)
    else:
        pow_vec = discount ** torch.arange(1, chunk_size + 1, device=device, dtype=dtype)

    rewards_flat = rewards_chunks.reshape(B * n_chunks, chunk_size)
    S_local_flat = rewards_flat @ M
    S_local_chunks = S_local_flat.view(B, n_chunks, chunk_size)

    lengths = [chunk_size] * n_chunks
    if pad > 0:
        lengths[-1] = chunk_size - pad

    S_rev = rewards_rev.new_zeros(B, T_pad)
    s_prev = torch.zeros(B, device=device, dtype=dtype)

    for c in range(n_chunks):
        Lc = lengths[c]
        start = c * chunk_size
        end = start + Lc

        S_local = S_local_chunks[:, c, :Lc]
        S_global = S_local + s_prev.unsqueeze(1) * pow_vec[:Lc]

        S_rev[:, start:end] = S_global
        s_prev = S_global[:, -1]

    if pad > 0:
        S_rev = S_rev[:, :T]

    return torch.flip(S_rev, dims=[1])


def chunked_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd: float,
    chunk_size: int = 128,
):
    """Compute Generalized Advantage Estimation using a chunked scan."""
    assert rewards.ndim == 2 and values.ndim == 2
    B, T = rewards.shape
    assert values.shape == (B, T)

    next_values = torch.cat(
        [values[:, 1:], torch.zeros(B, 1, device=values.device, dtype=values.dtype)],
        dim=1,
    )
    deltas = rewards + gamma * next_values - values
    advantages = chunked_discounted_returns(deltas, gamma * lambd, chunk_size)
    returns = advantages + values

    return advantages, returns


def calculate_log_probs_and_entropy(
    logits,
    tokens,
    tp_group,
    with_entropy: bool = False,
    chunk_size: int = -1,
    log_prob_keep_mask=None,
    with_entropy_grad: bool = True,
    with_dppo_directional_moment: bool = False,
):
    if with_dppo_directional_moment and not with_entropy:
        raise ValueError("DPPO entropy directional moments require with_entropy=True.")
    if with_dppo_directional_moment and log_prob_keep_mask is not None:
        raise ValueError("DPPO entropy directional moments do not support truncated top-p log-prob replay.")
    logits = logits.contiguous()
    entropy = None
    dppo_directional_moment = None
    if logits.size(0) != 0:
        if chunk_size > 0:
            num_chunks = (logits.size(0) - 1) // chunk_size + 1
            logits_chunks = logits.chunk(num_chunks, dim=0)
            tokens_chunks = tokens.chunk(num_chunks, dim=0)
            mask_chunks = (
                log_prob_keep_mask.chunk(num_chunks, dim=0) if log_prob_keep_mask is not None else [None] * num_chunks
            )

            if with_dppo_directional_moment:
                entropys = []
                dppo_directional_moments = []
                for logits_chunk in logits_chunks:
                    entropy_input = logits_chunk.clone() if with_entropy_grad else logits_chunk.detach().clone()
                    entropy_chunk, moment_chunk = compute_entropy_and_dppo_directional_moment_from_logits(
                        entropy_input, tp_group
                    )
                    entropys.append(entropy_chunk)
                    dppo_directional_moments.append(moment_chunk)
                entropy = torch.cat(entropys, dim=0)
                dppo_directional_moment = torch.cat(dppo_directional_moments, dim=0)

            log_probs = []
            entropy_chunks = []
            for tokens_chunk, logits_chunk, mask_chunk in zip(tokens_chunks, logits_chunks, mask_chunks, strict=True):
                if with_dppo_directional_moment:
                    log_prob = compute_log_probs(logits_chunk.clone(), tokens_chunk, tp_group)
                    entropy_chunk = None
                else:
                    log_prob, entropy_chunk = _calculate_log_probs_and_entropy_chunk(
                        logits_chunk,
                        tokens_chunk,
                        tp_group,
                        with_entropy=with_entropy,
                        with_entropy_grad=with_entropy_grad,
                        log_prob_keep_mask=mask_chunk,
                    )
                log_probs.append(log_prob)
                if entropy_chunk is not None:
                    entropy_chunks.append(entropy_chunk)
            log_prob = torch.cat(log_probs, dim=0)
            if entropy_chunks:
                entropy = torch.cat(entropy_chunks, dim=0)
        else:
            if with_dppo_directional_moment:
                entropy_input = logits.clone() if with_entropy_grad else logits.detach().clone()
                entropy, dppo_directional_moment = compute_entropy_and_dppo_directional_moment_from_logits(
                    entropy_input, tp_group
                )
                log_prob = compute_log_probs(logits.clone(), tokens, tp_group)
            else:
                log_prob, entropy = _calculate_log_probs_and_entropy_chunk(
                    logits,
                    tokens,
                    tp_group,
                    with_entropy=with_entropy,
                    with_entropy_grad=with_entropy_grad,
                    log_prob_keep_mask=log_prob_keep_mask,
                )
    else:
        log_prob = logits.new_zeros((0,))
        if with_entropy:
            entropy = logits.new_zeros((0,))
        if with_dppo_directional_moment:
            dppo_directional_moment = logits.new_zeros((0,))

    if with_dppo_directional_moment:
        return log_prob, entropy, dppo_directional_moment
    return log_prob, entropy
