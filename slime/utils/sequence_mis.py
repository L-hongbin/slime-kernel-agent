"""Sequence MIS admission shared by pre-training and live-forward callers.

Callers provide full responses (gather context-parallel shards beforehand).
This module neither selects the actor snapshot nor mutates the input masks.
"""

import math
from argparse import Namespace
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence

MISMATCH_EXCEED_THRESHOLDS = (0.02, 0.05)


@dataclass
class SequenceMISResult:
    loss_masks: list[torch.Tensor]
    keep: torch.Tensor
    scores: torch.Tensor
    stats: dict[str, float]


@torch.no_grad()
def compute_sequence_mis(
    args: Namespace,
    actor_log_probs: list[torch.Tensor],
    reference_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    advantages: list[torch.Tensor] | None = None,
    *,
    collect_stats: bool = False,
) -> SequenceMISResult:
    """Compute detached sample gates, scores and masks for complete responses.

    ``turns_*`` expects trajectory-major, complete groups of ``max_turns``.
    Token veto and positive-advantage protection remain per turn. ``loop`` and
    ``batch`` use the same math with different chunk sizes. Host-side diagnostic
    statistics are optional, so the live loss need not synchronize to read them.
    """
    aggregation = getattr(args, "sequence_mis_aggregation", None)
    if aggregation not in {None, "kl", "geometric", "mirrorpop", "turns_geometric", "turns_mirrorpop", "binary_kl"}:
        raise ValueError(f"Unknown Sequence MIS aggregation: {aggregation!r}")
    n = len(loss_masks)
    if len(actor_log_probs) != n or len(reference_log_probs) != n:
        raise ValueError("Sequence MIS log-probability and loss-mask list lengths must match.")
    use_advantage = bool(getattr(args, "sequence_mis_use_advantage", False))
    if use_advantage and (advantages is None or len(advantages) != n):
        raise ValueError("Sequence MIS use_advantage requires advantages for every sample.")
    lower = getattr(args, "sequence_mis_lower", None)
    upper = getattr(args, "sequence_mis_upper", None)
    lower = float("-inf") if lower is None else float(lower)
    upper = (0.05 if aggregation == "binary_kl" else float("inf")) if upper is None else float(upper)
    if lower >= upper:
        raise ValueError("Sequence MIS lower must be smaller than upper.")
    if aggregation == "binary_kl" and (not math.isfinite(upper) or upper < 0):
        raise ValueError("Binary KL sample gate threshold must be finite and nonnegative.")
    veto = getattr(args, "sequence_mis_token_veto_threshold", None)
    if veto is not None and (not math.isfinite(veto) or veto <= 0):
        raise ValueError("Sequence MIS token veto threshold must be positive and finite.")
    if aggregation == "binary_kl" and (lower != float("-inf") or veto is not None or use_advantage):
        raise ValueError(
            "Sequence MIS binary_kl only supports upper; token veto and advantage protection are disabled."
        )
    turns = aggregation in {"turns_geometric", "turns_mirrorpop"}
    max_turns = getattr(args, "max_turns", None) if turns else 1
    if max_turns is None or max_turns <= 0 or n % max_turns:
        raise ValueError("Sequence MIS turns_* requires max_turns and complete trajectory-major groups.")
    mode = getattr(args, "sequence_mis_mode", "batch") or "batch"
    if mode not in {"loop", "batch"}:
        raise ValueError("Sequence MIS mode must be 'loop' or 'batch'.")
    chunk_size = int(getattr(args, "sequence_mis_batch_size", 8) or 8) if mode == "batch" else 1
    if chunk_size <= 0:
        raise ValueError("Sequence MIS batch size must be positive.")
    chunk_size = ((chunk_size + max_turns - 1) // max_turns) * max_turns
    stats = dict.fromkeys(
        [
            "rejected",
            "valid_sequences",
            "ratio_sum",
            "advantage_protected",
            "abs_log_ratio_sum",
            "valid_tokens",
            "abs_log_ratio_max",
            *[f"tok_exceed_{t}" for t in MISMATCH_EXCEED_THRESHOLDS],
        ],
        0.0,
    )
    stats.update(min_ratio=float("inf"), max_ratio=0.0)
    keep_chunks, score_chunks, output_masks = [], [], []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        masks = loss_masks[start:end]
        for actor, reference, mask in zip(
            actor_log_probs[start:end], reference_log_probs[start:end], masks, strict=True
        ):
            if actor.ndim != 1 or actor.shape != reference.shape or actor.shape != mask.shape:
                raise ValueError(
                    "Sequence MIS requires matching one-dimensional log-probability and loss-mask shapes."
                )
        mask = pad_sequence([m.float() for m in masks], batch_first=True)
        valid = mask.bool()
        actor = pad_sequence([p.float() for p in actor_log_probs[start:end]], batch_first=True)
        reference = pad_sequence([p.float() for p in reference_log_probs[start:end]], batch_first=True)
        actor = torch.where(valid, actor, 0.0)
        reference = torch.where(valid, reference, 0.0)
        ratios = (actor - reference) * mask
        counts = mask.sum(dim=1)
        valid_sequences = counts > 0
        finite = torch.isfinite(ratios).all(dim=1)
        if aggregation == "binary_kl":
            p, q = reference.exp().clamp(1e-6, 1 - 1e-6), actor.exp().clamp(1e-6, 1 - 1e-6)
            divergence = p * (p.log() - q.log()) + (1 - p) * (torch.log1p(-p) - torch.log1p(-q))
            scores = (torch.where(valid, divergence, 0.0).sum(dim=1) / valid.sum(dim=1).clamp_min(1)).clamp_min(0)
            keep = finite & (actor <= 0).all(dim=1) & (reference <= 0).all(dim=1) & (scores <= upper)
        else:
            sums = ratios.abs().sum(dim=1) if aggregation in {"mirrorpop", "turns_mirrorpop"} else ratios.sum(dim=1)
            if turns:
                scores = sums.reshape(-1, max_turns).sum(dim=1) / counts.reshape(-1, max_turns).sum(dim=1).clamp_min(1)
                scores = scores.repeat_interleave(max_turns)
                finite = finite.reshape(-1, max_turns).all(dim=1).repeat_interleave(max_turns)
            else:
                scores = sums / counts.clamp_min(1)
            if aggregation in {"geometric", "turns_geometric"}:
                scores = scores.clamp(-20, 20).exp()
            elif aggregation == "kl":
                scores = -scores
            elif aggregation is None:
                scores = torch.zeros_like(scores)
            keep = finite
            if aggregation is not None:
                keep = keep & (scores >= lower) & (scores <= upper)
            if veto is not None:
                keep = keep & ~((ratios < math.log(veto)) & valid).any(dim=1)
        protected = torch.zeros_like(keep)
        if use_advantage:
            protected = torch.stack(
                [
                    (((a > 0) & m.bool()) if a.shape == m.shape else (a > 0)).any()
                    for a, m in zip(advantages[start:end], masks, strict=True)
                ]
            )
            keep = keep | protected
        keep_chunks.append(keep)
        score_chunks.append(scores)
        output_masks.extend(m * k.to(m.dtype) for m, k in zip(masks, keep.unbind(), strict=True))
        if collect_stats:
            stats["rejected"] += float((valid_sequences & ~keep).sum().item())
            stats["valid_sequences"] += float(valid_sequences.sum().item())
            stats["advantage_protected"] += float((valid_sequences & protected).sum().item())
            if aggregation is not None and valid_sequences.any():
                values = scores[valid_sequences]
                stats["ratio_sum"] += float(values.sum().item())
                stats["min_ratio"] = min(stats["min_ratio"], float(values.min().item()))
                stats["max_ratio"] = max(stats["max_ratio"], float(values.max().item()))
            stats["valid_tokens"] += float(counts.sum().item())
            stats["abs_log_ratio_sum"] += float(ratios.abs().sum().item())
            if ratios.numel():
                stats["abs_log_ratio_max"] = max(stats["abs_log_ratio_max"], float(ratios.abs().max().item()))
            for threshold in MISMATCH_EXCEED_THRESHOLDS:
                stats[f"tok_exceed_{threshold}"] += float((ratios.abs() > threshold).sum().item())
    return SequenceMISResult(
        loss_masks=output_masks,
        keep=torch.cat(keep_chunks) if keep_chunks else torch.empty(0, dtype=torch.bool),
        scores=torch.cat(score_chunks) if score_chunks else torch.empty(0),
        stats=stats if collect_stats else {},
    )
