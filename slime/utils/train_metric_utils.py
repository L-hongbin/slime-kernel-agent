import logging
import math
from argparse import Namespace
from collections.abc import Callable
from copy import deepcopy

from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step
from slime.utils.timer import Timer

logger = logging.getLogger(__name__)


ENTROPY_COMMON_PROBE_MASK_KEY = "entropy_common_probe_loss_masks"
ENTROPY_COMMON_PROBE_NUMERATOR_KEY = "_entropy_common_probe_numerator"
ENTROPY_COMMON_PROBE_DENOMINATOR_KEY = "_entropy_common_probe_denominator"
ENTROPY_COMMON_PROBE_METRIC_KEY = "entropy_common_probe"

ENTROPY_GROUPS = (
    "adv_positive_ratio_ge_1",
    "adv_positive_ratio_lt_1",
    "adv_negative_ratio_ge_1",
    "adv_negative_ratio_lt_1",
    "upper_clipped",
    "lower_clipped",
)


def add_entropy_common_probe_metric(metrics: dict[str, float], *, required: bool = False) -> dict[str, float]:
    """Finalize the fixed-batch entropy probe after DP×CP reduction.

    The loss path emits a masked entropy sum and its masked token count as
    separate linearly reducible values.  ``reduce_train_step_metrics`` applies
    the same outer normalization to both, so their ratio is the exact global
    token mean regardless of whether the training loss itself uses token or
    rollout normalization.  Raw sufficient statistics are intentionally
    removed before logging.
    """
    numerator_present = ENTROPY_COMMON_PROBE_NUMERATOR_KEY in metrics
    denominator_present = ENTROPY_COMMON_PROBE_DENOMINATOR_KEY in metrics
    if not numerator_present and not denominator_present:
        if required:
            raise RuntimeError(
                "--entropy-common-probe is enabled but the reduced train metrics contain no "
                "common-probe sufficient statistics"
            )
        return metrics
    if numerator_present != denominator_present:
        missing = ENTROPY_COMMON_PROBE_DENOMINATOR_KEY if numerator_present else ENTROPY_COMMON_PROBE_NUMERATOR_KEY
        raise RuntimeError(f"entropy common-probe reduction is missing {missing}")

    numerator = float(metrics[ENTROPY_COMMON_PROBE_NUMERATOR_KEY])
    denominator = float(metrics[ENTROPY_COMMON_PROBE_DENOMINATOR_KEY])
    if not math.isfinite(numerator):
        raise RuntimeError(f"entropy common-probe numerator must be finite, got {numerator}")
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError(f"entropy common-probe denominator must be finite and positive, got {denominator}")

    metrics.pop(ENTROPY_COMMON_PROBE_NUMERATOR_KEY)
    metrics.pop(ENTROPY_COMMON_PROBE_DENOMINATOR_KEY)
    metrics[ENTROPY_COMMON_PROBE_METRIC_KEY] = numerator / denominator
    return metrics


def format_train_metric_key(key: str, role_tag: str = "") -> str:
    """Map reduced train metrics to their tracking namespace.

    DPPO diagnostics are intentionally a top-level dashboard group.  Every
    other loss/optimizer metric keeps slime's historical ``train/`` prefix.
    ``role_tag`` is inserted after the namespace so non-actor roles cannot
    collide if they ever emit the same diagnostic.
    """
    if key == "entropy_loss":
        return f"entropy/{role_tag}train"
    if key == ENTROPY_COMMON_PROBE_METRIC_KEY:
        return f"entropy/{role_tag}common_probe"
    if key.startswith("entropy/"):
        return f"entropy/{role_tag}{key.removeprefix('entropy/')}"
    if key.startswith("dppo/"):
        return f"dppo/{role_tag}{key.removeprefix('dppo/')}"
    return f"train/{role_tag}{key}"


def add_derived_entropy_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Turn globally reduced entropy joint moments into conditional means."""
    for group in ENTROPY_GROUPS:
        fraction_key = f"_entropy/{group}_fraction"
        joint_key = f"_entropy/{group}_joint_mean"
        if fraction_key not in metrics and joint_key not in metrics:
            continue
        if fraction_key not in metrics or joint_key not in metrics:
            missing = joint_key if fraction_key in metrics else fraction_key
            raise RuntimeError(f"entropy metric reduction is missing {missing}")
        fraction = float(metrics.pop(fraction_key))
        joint_mean = float(metrics.pop(joint_key))
        metrics[f"entropy/{group}"] = joint_mean / fraction if fraction > 0.0 else 0.0
    return metrics


def add_derived_dppo_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Derive DPPO conditional diagnostics after global DP×CP reduction.

    The loss path reports only linearly reducible joint moments.  Ratios formed
    inside a microbatch would be biased when a microbatch contains only one
    advantage sign, so all divisions happen here after ``train_one_step`` has
    reduced the sufficient statistics over the complete step.
    """
    if "dppo/adv_positive_token_frac" not in metrics:
        return metrics

    def ratio(numerator_key: str, denominator_key: str) -> float:
        denominator = float(metrics[denominator_key])
        return float(metrics[numerator_key]) / denominator if denominator > 0.0 else 0.0

    metrics["dppo/upper_clip_rate_given_positive"] = ratio(
        "dppo/upper_clip_joint_frac", "dppo/adv_positive_token_frac"
    )
    metrics["dppo/lower_clip_rate_given_negative"] = ratio(
        "dppo/lower_clip_joint_frac", "dppo/adv_negative_token_frac"
    )
    metrics["dppo/masked_update_mass_positive_frac"] = ratio(
        "dppo/masked_update_mass_positive_mean", "dppo/update_mass_positive_mean"
    )
    metrics["dppo/masked_update_mass_negative_frac"] = ratio(
        "dppo/masked_update_mass_negative_mean", "dppo/update_mass_negative_mean"
    )

    unmasked_abs_mass = float(metrics["dppo/update_mass_positive_mean"]) + float(
        metrics["dppo/update_mass_negative_mean"]
    )
    kept_abs_mass = float(metrics["dppo/kept_update_mass_mean"])
    unmasked_push = (
        float(metrics["dppo/net_logprob_push_unmasked_numerator_mean"]) / unmasked_abs_mass
        if unmasked_abs_mass > 0.0
        else 0.0
    )
    kept_push = (
        float(metrics["dppo/net_logprob_push_kept_numerator_mean"]) / kept_abs_mass if kept_abs_mass > 0.0 else 0.0
    )
    metrics["dppo/net_logprob_push_unmasked"] = unmasked_push
    metrics["dppo/net_logprob_push_kept"] = kept_push
    metrics["dppo/mask_induced_push_delta"] = kept_push - unmasked_push

    category_denominators = {
        "positive_clipped": "dppo/upper_clip_joint_frac",
        "positive_kept": "dppo/positive_kept_joint_frac",
        "negative_clipped": "dppo/lower_clip_joint_frac",
        "negative_kept": "dppo/negative_kept_joint_frac",
    }
    for category, denominator_key in category_denominators.items():
        for value_name in ("train_sampled_prob", "rollout_sampled_prob"):
            joint_key = f"dppo/{value_name}_{category}_joint_mean"
            metrics[f"dppo/{value_name}_{category}"] = ratio(joint_key, denominator_key)

    return metrics


def log_perf_data_raw(
    rollout_id: int,
    args: Namespace,
    is_primary_rank: bool,
    compute_total_fwd_flops: Callable,
    extra_metrics: dict | None = None,
) -> None:
    timer_instance = Timer()
    log_dict_raw = deepcopy(timer_instance.log_dict())
    timer_instance.reset()

    if not is_primary_rank:
        return

    log_dict = {f"perf/{key}_time": val for key, val in log_dict_raw.items()}
    if extra_metrics:
        log_dict.update(extra_metrics)

    if ("perf/actor_train_time" in log_dict) and (compute_total_fwd_flops is not None):
        total_fwd_flops = compute_total_fwd_flops(seq_lens=timer_instance.seq_lens)

        if "perf/log_probs_time" in log_dict:
            log_dict["perf/log_probs_tflops"] = total_fwd_flops / log_dict["perf/log_probs_time"]

        if "perf/ref_log_probs_time" in log_dict:
            log_dict["perf/ref_log_probs_tflops"] = total_fwd_flops / log_dict["perf/ref_log_probs_time"]

        if log_dict["perf/actor_train_time"] > 0:
            log_dict["perf/actor_train_tflops"] = 3 * total_fwd_flops / log_dict["perf/actor_train_time"]
            log_dict["perf/actor_train_tok_per_s"] = sum(timer_instance.seq_lens) / log_dict["perf/actor_train_time"]

    if "perf/train_wait_time" in log_dict and "perf/train_time" in log_dict:
        total_time = log_dict["perf/train_wait_time"] + log_dict["perf/train_time"]
        if total_time > 0:
            log_dict["perf/step_time"] = total_time
            log_dict["perf/wait_time_ratio"] = log_dict["perf/train_wait_time"] / total_time

    logger.info(f"perf {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    if args.wandb_always_use_train_step:
        # Both axes share train-step units; carry both keys so a single
        # workspace-wide x-axis works for every metric section.
        log_dict["train/step"] = step
    logging_utils.log(args, log_dict, step_key="rollout/step")


def log_named_perf_timers_raw(
    *,
    rollout_id: int,
    args: Namespace,
    is_primary_rank: bool,
    timer_names: tuple[str, ...],
) -> None:
    """Pop and immediately report selected timers at their owning rollout.

    Checkpoint operations run after the normal train-step perf flush. Leaving
    their timers in the singleton would attribute them to the next rollout and
    lose the final rollout entirely. Pop only the requested names so unrelated
    train/update timers keep their existing lifecycle.
    """
    timer_instance = Timer()
    selected = {name: timer_instance.timers.pop(name) for name in timer_names if name in timer_instance.timers}
    if not is_primary_rank or not selected:
        return

    metrics = {f"perf/{name}_time": value for name, value in selected.items()}
    logger.info("perf checkpoint %s: %s", rollout_id, metrics)
    step = compute_rollout_step(args, rollout_id)
    metrics["rollout/step"] = step
    if args.wandb_always_use_train_step:
        metrics["train/step"] = step
    try:
        logging_utils.log(args, metrics, step_key="rollout/step")
    except Exception:
        # Performance reporting must never strand a checkpoint request between
        # actor-side scheduling and driver-side lifecycle registration.
        logger.exception("Failed to report checkpoint perf metrics for rollout %s", rollout_id)
