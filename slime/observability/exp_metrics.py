"""Low-overhead experimental diagnostics for Binary-DPPO.

The helpers in this module only consume tensors already materialized by the
policy loss.  They emit scalar sufficient statistics; callers perform the
existing DP/CP reduction before deriving conditional ratios.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

EXP_PREFIX = "exp/"
_PRIVATE_EXP_PREFIX = "_exp/"

DPPO_TV_THRESHOLDS = (
    ("0.02", 0.02),
    ("0.05", 0.05),
    ("0.10", 0.10),
    ("0.15", 0.15),
    ("0.20", 0.20),
    ("0.30", 0.30),
)

DPPO_BEHAVIOR_PROB_BINS = (
    ("0_1e-3", 0.0, 1e-3),
    ("1e-3_1e-2", 1e-3, 1e-2),
    ("1e-2_0.1", 1e-2, 0.1),
    ("0.1_0.2", 0.1, 0.2),
    ("0.2_0.5", 0.2, 0.5),
    ("0.5_0.8", 0.5, 0.8),
    ("0.8_0.95", 0.8, 0.95),
    ("0.95_1", 0.95, 1.0),
)

DPPO_RELATIVE_MOVE_THRESHOLDS = (0.25, 0.50, 0.90)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def compute_binary_dppo_exp_metrics(
    *,
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mode: str,
    eps_clip: float,
    eps_clip_high: float,
    ratio_clip_c: float | None,
    metric_reducer: Callable[[torch.Tensor], torch.Tensor],
    policy_lags: torch.Tensor | None = None,
    sample_ages_seconds: torch.Tensor | None = None,
    turn_indices: torch.Tensor | None = None,
    engine_version_spans: torch.Tensor | None = None,
    engine_version_mismatches: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return reduced Binary-DPPO sufficient statistics.

    All conditionals are represented as joint numerators and denominators.  No
    microbatch-local ratio is formed here, so the final result remains correct
    after the regular DP/CP all-reduce.
    """

    if log_probs.shape != old_log_probs.shape or log_probs.shape != advantages.shape:
        raise ValueError(
            "Binary-DPPO exp metrics require aligned log_probs, old_log_probs, and advantages; "
            f"got {tuple(log_probs.shape)}, {tuple(old_log_probs.shape)}, {tuple(advantages.shape)}"
        )

    with torch.no_grad():
        calc_dtype = torch.float32
        current_log_prob = log_probs.detach().to(calc_dtype)
        behavior_log_prob = old_log_probs.detach().to(calc_dtype)
        advantage = advantages.detach().to(calc_dtype)
        current_prob = current_log_prob.exp()
        behavior_prob = behavior_log_prob.exp()
        prob_delta = current_prob - behavior_prob
        abs_prob_delta = prob_delta.abs()
        log_ratio = (current_log_prob - behavior_log_prob).clamp(min=-20.0, max=20.0)
        abs_log_ratio = log_ratio.abs()
        importance_ratio = log_ratio.exp()
        importance_weight = importance_ratio.clamp(max=20.0 if ratio_clip_c is None else float(ratio_clip_c))

        tiny = torch.finfo(calc_dtype).tiny
        current_other = (1.0 - current_prob).clamp_min(tiny)
        behavior_other = (1.0 - behavior_prob).clamp_min(tiny)
        binary_kl_raw = behavior_prob * (behavior_log_prob - current_log_prob) + behavior_other * (
            behavior_other.log() - current_other.log()
        )
        binary_kl = binary_kl_raw.clamp_min(0.0)

        positive = advantage > 0
        negative = advantage < 0
        zero = advantage == 0
        if loss_mode == "dppo_binary_tv":
            invalid_positive = prob_delta > float(eps_clip_high)
            invalid_negative = prob_delta < -float(eps_clip)
        elif loss_mode == "dppo_binary_kl":
            invalid_positive = (binary_kl > float(eps_clip_high)) & (prob_delta > 0)
            invalid_negative = (binary_kl > float(eps_clip)) & (prob_delta < 0)
        else:
            raise ValueError(f"unsupported Binary-DPPO exp metric mode: {loss_mode}")
        upper_clipped = positive & invalid_positive
        lower_clipped = negative & invalid_negative
        zero_lower_clipped = zero & invalid_negative
        clipped = upper_clipped | lower_clipped
        kept = ~clipped
        update_mass = advantage.abs() * importance_weight
        relative_positive_move = prob_delta.clamp_min(0.0) / (1.0 - behavior_prob).clamp_min(tiny)
        relative_negative_move = (-prob_delta).clamp_min(0.0) / behavior_prob.clamp_min(tiny)

        metrics: dict[str, torch.Tensor] = {}

        def add(name: str, value: torch.Tensor) -> None:
            metrics[f"{_PRIVATE_EXP_PREFIX}train/dppo/{name}"] = metric_reducer(value.to(calc_dtype)).detach()

        add("importance_ratio/mean", importance_ratio)
        add("importance_weight/mean", importance_weight)
        add("binary_tv/mean", abs_prob_delta)
        add("binary_kl/mean", binary_kl)
        add("binary_kl/numerical_negative_fraction", (binary_kl_raw < 0).to(calc_dtype))
        add("abs_log_ratio/mean", abs_log_ratio)

        sign_data = {
            "positive": (positive, upper_clipped, prob_delta > 0, relative_positive_move),
            "negative": (negative, lower_clipped, prob_delta < 0, relative_negative_move),
        }
        add("adv/zero_token_fraction", zero.to(calc_dtype))
        add("clip/zero_lower_joint_fraction", zero_lower_clipped.to(calc_dtype))

        for sign, (indicator, sign_clipped, outward, relative_move) in sign_data.items():
            indicator_float = indicator.to(calc_dtype)
            add(f"adv/{sign}_token_fraction", indicator_float)
            add(f"adv/{sign}_abs_joint_mean", advantage.abs() * indicator_float)
            add(f"clip/{sign}_joint_fraction", sign_clipped.to(calc_dtype))
            add(f"outward/{sign}_joint_fraction", (indicator & outward).to(calc_dtype))
            add(f"update_mass/{sign}_joint_mean", update_mass * indicator_float)
            add(f"masked_update_mass/{sign}_joint_mean", update_mass * sign_clipped)
            add(f"kept_update_mass/{sign}_joint_mean", update_mass * (indicator & kept))
            add(f"prob_delta/{sign}_joint_mean", prob_delta * indicator_float)
            add(f"abs_prob_delta/{sign}_joint_mean", abs_prob_delta * indicator_float)
            add(f"abs_log_ratio/{sign}_joint_mean", abs_log_ratio * indicator_float)
            add(f"binary_kl/{sign}_joint_mean", binary_kl * indicator_float)
            add(f"relative_move/{sign}_joint_mean", relative_move * indicator_float)
            for threshold in DPPO_RELATIVE_MOVE_THRESHOLDS:
                label = f"{int(threshold * 100)}pct"
                add(
                    f"relative_move/{sign}_gt_{label}_joint_fraction",
                    (indicator & (relative_move > threshold)).to(calc_dtype),
                )

        for label, threshold in DPPO_TV_THRESHOLDS:
            if loss_mode == "dppo_binary_tv":
                positive_clip = positive & (prob_delta > threshold)
                negative_clip = negative & (prob_delta < -threshold)
            else:
                positive_clip = positive & (binary_kl > threshold) & (prob_delta > 0)
                negative_clip = negative & (binary_kl > threshold) & (prob_delta < 0)
            for sign, indicator in (("positive", positive_clip), ("negative", negative_clip)):
                add(f"delta/{label}/{sign}_clip_joint_fraction", indicator.to(calc_dtype))
                add(f"delta/{label}/{sign}_masked_mass_joint_mean", update_mass * indicator)

        for label, lower, upper in DPPO_BEHAVIOR_PROB_BINS:
            in_bin = (behavior_prob >= lower) & (behavior_prob <= upper if upper == 1.0 else behavior_prob < upper)
            for sign, (sign_indicator, sign_clipped, _outward, relative_move) in sign_data.items():
                indicator = in_bin & sign_indicator
                indicator_float = indicator.to(calc_dtype)
                clipped_indicator = indicator & sign_clipped
                prefix = f"behavior_prob/{label}/{sign}"
                add(f"{prefix}/token_joint_fraction", indicator_float)
                add(f"{prefix}/update_mass_joint_mean", update_mass * indicator_float)
                add(f"{prefix}/clip_joint_fraction", clipped_indicator.to(calc_dtype))
                add(f"{prefix}/masked_mass_joint_mean", update_mass * clipped_indicator)
                add(f"{prefix}/abs_prob_delta_joint_mean", abs_prob_delta * indicator_float)
                add(f"{prefix}/abs_log_ratio_joint_mean", abs_log_ratio * indicator_float)
                add(f"{prefix}/binary_kl_joint_mean", binary_kl * indicator_float)
                add(f"{prefix}/relative_move_joint_mean", relative_move * indicator_float)

        def add_context_metrics(prefix: str, buckets: tuple[tuple[str, torch.Tensor], ...]) -> None:
            for label, context_indicator in buckets:
                context_indicator = context_indicator.bool()
                context_float = context_indicator.to(calc_dtype)
                add(f"{prefix}/{label}/token_fraction", context_float)
                add(f"{prefix}/{label}/binary_tv_joint_mean", abs_prob_delta * context_float)
                add(f"{prefix}/{label}/binary_kl_joint_mean", binary_kl * context_float)
                add(f"{prefix}/{label}/abs_log_ratio_joint_mean", abs_log_ratio * context_float)
                for sign, (sign_indicator, sign_clipped, _outward, _relative_move) in sign_data.items():
                    indicator = context_indicator & sign_indicator
                    clipped_indicator = context_indicator & sign_clipped
                    add(f"{prefix}/{label}/{sign}_token_joint_fraction", indicator.to(calc_dtype))
                    add(f"{prefix}/{label}/{sign}_clip_joint_fraction", clipped_indicator.to(calc_dtype))
                    add(
                        f"{prefix}/{label}/{sign}_masked_mass_joint_mean",
                        update_mass * clipped_indicator,
                    )
                    add(f"{prefix}/{label}/{sign}_update_mass_joint_mean", update_mass * indicator)

        if policy_lags is not None:
            policy_lags = policy_lags.to(device=log_probs.device)
            if policy_lags.shape != log_probs.shape:
                raise ValueError("policy_lags must align with Binary-DPPO token tensors")
            valid_lag = torch.isfinite(policy_lags)
            add("async/policy_lag_valid_fraction", valid_lag.to(calc_dtype))
            add("async/policy_lag_joint_mean", torch.where(valid_lag, policy_lags, 0.0))
            lag_buckets = tuple(
                [(str(lag), valid_lag & (policy_lags == lag)) for lag in range(4)]
                + [("ge4", valid_lag & (policy_lags >= 4))]
            )
            add_context_metrics("async/lag", lag_buckets)

        if sample_ages_seconds is not None:
            sample_ages_seconds = sample_ages_seconds.to(device=log_probs.device)
            if sample_ages_seconds.shape != log_probs.shape:
                raise ValueError("sample_ages_seconds must align with Binary-DPPO token tensors")
            valid_age = torch.isfinite(sample_ages_seconds)
            add("async/sample_age_valid_fraction", valid_age.to(calc_dtype))
            add("async/sample_age_seconds_joint_mean", torch.where(valid_age, sample_ages_seconds, 0.0))
            age_buckets = (
                ("lt60s", valid_age & (sample_ages_seconds < 60)),
                ("60_300s", valid_age & (sample_ages_seconds >= 60) & (sample_ages_seconds < 300)),
                ("300_900s", valid_age & (sample_ages_seconds >= 300) & (sample_ages_seconds < 900)),
                ("ge900s", valid_age & (sample_ages_seconds >= 900)),
            )
            add_context_metrics("async/age", age_buckets)

        for name, values in (
            ("engine_version_span_fraction", engine_version_spans),
            ("engine_version_mismatch_fraction", engine_version_mismatches),
        ):
            if values is None:
                continue
            values = values.to(device=log_probs.device)
            if values.shape != log_probs.shape:
                raise ValueError(f"{name} must align with Binary-DPPO token tensors")
            add(f"async/{name}", values)

        if turn_indices is not None:
            turn_indices = turn_indices.to(device=log_probs.device)
            if turn_indices.shape != log_probs.shape:
                raise ValueError("turn_indices must align with Binary-DPPO token tensors")
            valid_turn = torch.isfinite(turn_indices)
            turn_buckets = tuple(
                [(str(turn), valid_turn & (turn_indices == turn)) for turn in range(3)]
                + [("ge3", valid_turn & (turn_indices >= 3))]
            )
            add_context_metrics("turn", turn_buckets)

    return metrics


def finalize_exp_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Publish private sufficient statistics and add conditional ratios."""

    private = {key: float(value) for key, value in metrics.items() if key.startswith(_PRIVATE_EXP_PREFIX)}
    if not private:
        return metrics
    for key in private:
        metrics.pop(key)
    public = {key.removeprefix("_"): value for key, value in private.items()}
    # ``*_joint_*`` values are sufficient statistics used only to form a
    # globally correct conditional ratio.  Keep them out of user-facing logs
    # and TensorBoard to avoid duplicating every useful metric with its
    # numerator/denominator implementation detail.
    metrics.update({key: value for key, value in public.items() if "_joint_" not in key})

    base = "exp/train/dppo"
    for sign in ("positive", "negative"):
        sign_fraction = public.get(f"{base}/adv/{sign}_token_fraction", 0.0)
        sign_mass = public.get(f"{base}/update_mass/{sign}_joint_mean", 0.0)
        metrics[f"{base}/clip/{sign}_rate"] = _safe_ratio(
            public.get(f"{base}/clip/{sign}_joint_fraction", 0.0), sign_fraction
        )
        metrics[f"{base}/outward/{sign}_rate"] = _safe_ratio(
            public.get(f"{base}/outward/{sign}_joint_fraction", 0.0), sign_fraction
        )
        metrics[f"{base}/masked_update_mass/{sign}_fraction"] = _safe_ratio(
            public.get(f"{base}/masked_update_mass/{sign}_joint_mean", 0.0), sign_mass
        )
        for name in ("abs_prob_delta", "abs_log_ratio", "binary_kl", "relative_move"):
            metrics[f"{base}/{name}/{sign}_mean"] = _safe_ratio(
                public.get(f"{base}/{name}/{sign}_joint_mean", 0.0), sign_fraction
            )
        for threshold in DPPO_RELATIVE_MOVE_THRESHOLDS:
            label = f"{int(threshold * 100)}pct"
            metrics[f"{base}/relative_move/{sign}_gt_{label}_rate"] = _safe_ratio(
                public.get(f"{base}/relative_move/{sign}_gt_{label}_joint_fraction", 0.0), sign_fraction
            )

        for label, _threshold in DPPO_TV_THRESHOLDS:
            metrics[f"{base}/delta/{label}/{sign}_clip_rate"] = _safe_ratio(
                public.get(f"{base}/delta/{label}/{sign}_clip_joint_fraction", 0.0), sign_fraction
            )
            metrics[f"{base}/delta/{label}/{sign}_masked_mass_fraction"] = _safe_ratio(
                public.get(f"{base}/delta/{label}/{sign}_masked_mass_joint_mean", 0.0), sign_mass
            )

        for label, _lower, _upper in DPPO_BEHAVIOR_PROB_BINS:
            prefix = f"{base}/behavior_prob/{label}/{sign}"
            bin_fraction = public.get(f"{prefix}/token_joint_fraction", 0.0)
            bin_mass = public.get(f"{prefix}/update_mass_joint_mean", 0.0)
            metrics[f"{prefix}/share_of_{sign}_tokens"] = _safe_ratio(bin_fraction, sign_fraction)
            metrics[f"{prefix}/share_of_{sign}_update_mass"] = _safe_ratio(bin_mass, sign_mass)
            metrics[f"{prefix}/clip_rate"] = _safe_ratio(
                public.get(f"{prefix}/clip_joint_fraction", 0.0), bin_fraction
            )
            metrics[f"{prefix}/masked_mass_fraction"] = _safe_ratio(
                public.get(f"{prefix}/masked_mass_joint_mean", 0.0), bin_mass
            )
            for name in ("abs_prob_delta", "abs_log_ratio", "binary_kl", "relative_move"):
                metrics[f"{prefix}/{name}_mean"] = _safe_ratio(
                    public.get(f"{prefix}/{name}_joint_mean", 0.0), bin_fraction
                )

    valid_lag_fraction = public.get(f"{base}/async/policy_lag_valid_fraction", 0.0)
    metrics[f"{base}/async/policy_lag_mean"] = _safe_ratio(
        public.get(f"{base}/async/policy_lag_joint_mean", 0.0), valid_lag_fraction
    )
    valid_age_fraction = public.get(f"{base}/async/sample_age_valid_fraction", 0.0)
    metrics[f"{base}/async/sample_age_seconds_mean"] = _safe_ratio(
        public.get(f"{base}/async/sample_age_seconds_joint_mean", 0.0), valid_age_fraction
    )

    for context in ("async/lag", "async/age", "turn"):
        context_prefix = f"{base}/{context}/"
        labels = {
            key.removeprefix(context_prefix).split("/", 1)[0] for key in public if key.startswith(context_prefix)
        }
        for label in labels:
            prefix = f"{context_prefix}{label}"
            fraction = public.get(f"{prefix}/token_fraction", 0.0)
            for name in ("binary_tv", "binary_kl", "abs_log_ratio"):
                metrics[f"{prefix}/{name}_mean"] = _safe_ratio(
                    public.get(f"{prefix}/{name}_joint_mean", 0.0), fraction
                )
            for sign in ("positive", "negative"):
                sign_fraction = public.get(f"{prefix}/{sign}_token_joint_fraction", 0.0)
                sign_mass = public.get(f"{prefix}/{sign}_update_mass_joint_mean", 0.0)
                metrics[f"{prefix}/{sign}_clip_rate"] = _safe_ratio(
                    public.get(f"{prefix}/{sign}_clip_joint_fraction", 0.0), sign_fraction
                )
                metrics[f"{prefix}/{sign}_masked_mass_fraction"] = _safe_ratio(
                    public.get(f"{prefix}/{sign}_masked_mass_joint_mean", 0.0), sign_mass
                )

    return metrics


def split_exp_metrics(metrics: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    """Return regular and ``exp/`` metrics without mutating the input."""

    regular = {key: value for key, value in metrics.items() if not key.startswith(EXP_PREFIX)}
    experimental = {key: value for key, value in metrics.items() if key.startswith(EXP_PREFIX)}
    return regular, experimental
