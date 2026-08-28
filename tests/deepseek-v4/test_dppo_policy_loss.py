"""Unit tests for compute_dppo_binary_policy_loss (Stable-RL DPPO recipe).

Reference semantics (user-provided core + Stable-RL core_algos.py):
  - binary-TV: mask positive-advantage tokens whose prob rose by > eps_high;
    mask negative-advantage tokens whose prob fell by > eps_low.
  - binary-KL: same directions, thresholded on the binary (2-outcome) KL.
  - surrogate: -A * detach(clamped IS ratio) * valid_mask * log_prob —
    gradient flows ONLY through log_probs.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slime.utils.ppo_utils import compute_dppo_binary_policy_loss  # noqa: E402

NUM_GPUS = 0


def _lp(p):
    return torch.log(torch.tensor(p, dtype=torch.float64))


def test_tv_masks_follow_advantage_direction():
    # probs: old 0.5 -> new {0.8 (+0.3), 0.45 (-0.05), 0.1 (-0.4)}
    log_probs = _lp([0.8, 0.45, 0.1]).requires_grad_(True)
    old = _lp([0.5, 0.5, 0.5])
    adv = torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64)
    out = compute_dppo_binary_policy_loss(log_probs, old, adv, 0.2, 0.2, "dppo_binary_tv")
    # token0: A>0, dprob=+0.3 > 0.2 -> masked (upper)
    # token1: A>0, dprob=-0.05, positive mask checks only upward moves -> valid
    # token2: A<0, dprob=-0.4 < -0.2 -> masked (lower)
    assert out["pg_clipfrac"].tolist() == [1.0, 0.0, 1.0]
    assert out["pg_upper_clipfrac"].tolist() == [1.0, 0.0, 0.0]
    assert out["pg_lower_clipfrac"].tolist() == [0.0, 0.0, 1.0]
    assert out["pg_losses"][0].item() == 0.0 and out["pg_losses"][2].item() == 0.0
    assert out["pg_losses"][1].item() != 0.0


def test_tv_asymmetric_bounds():
    # eps_clip (low, downward for A<0) = 0.1; eps_clip_high (upward for A>0) = 0.3
    log_probs = _lp([0.75, 0.35])
    old = _lp([0.5, 0.5])
    adv = torch.tensor([1.0, -1.0], dtype=torch.float64)
    out = compute_dppo_binary_policy_loss(log_probs, old, adv, 0.1, 0.3, "dppo_binary_tv")
    # token0: +0.25 <= 0.3 -> valid; token1: -0.15 < -0.1 -> masked
    assert out["pg_clipfrac"].tolist() == [0.0, 1.0]


def test_kl_mask_matches_manual_binary_kl():
    p_new, p_old = 0.9, 0.5
    log_probs = _lp([p_new])
    old = _lp([p_old])
    adv = torch.tensor([1.0], dtype=torch.float64)
    bkl = p_old * math.log(p_old / p_new) + (1 - p_old) * math.log((1 - p_old) / (1 - p_new))
    out_tight = compute_dppo_binary_policy_loss(log_probs, old, adv, bkl * 0.9, bkl * 0.9, "dppo_binary_kl")
    out_loose = compute_dppo_binary_policy_loss(log_probs, old, adv, bkl * 1.1, bkl * 1.1, "dppo_binary_kl")
    assert out_tight["pg_clipfrac"].item() == 1.0  # threshold below actual KL -> masked
    assert out_loose["pg_clipfrac"].item() == 0.0


def test_kl_direction_gating():
    # prob DECREASED with A>0: positive mask requires prob>old_prob -> stays valid
    log_probs = _lp([0.2])
    old = _lp([0.6])
    adv = torch.tensor([1.0], dtype=torch.float64)
    out = compute_dppo_binary_policy_loss(log_probs, old, adv, 1e-6, 1e-6, "dppo_binary_kl")
    assert out["pg_clipfrac"].item() == 0.0


def test_gradient_only_through_log_probs_and_ratio_detached():
    log_probs = _lp([0.6]).requires_grad_(True)
    old = _lp([0.5]).requires_grad_(True)
    adv = torch.tensor([2.0], dtype=torch.float64)
    out = compute_dppo_binary_policy_loss(log_probs, old, adv, 0.5, 0.5, "dppo_binary_tv")
    out["pg_losses"].sum().backward()
    ratio = 0.6 / 0.5
    # d/dlogp of (-A * detach(ratio) * logp) = -A * ratio
    assert torch.allclose(log_probs.grad, torch.tensor([-2.0 * ratio], dtype=torch.float64))
    assert old.grad is None or torch.all(old.grad == 0)


def test_ratio_cap():
    # ratio = e^{5} >> default cap 20 -> capped at 20; custom cap 2 -> 2
    log_probs = _lp([0.9])
    old = torch.tensor([math.log(0.9) - 5.0], dtype=torch.float64)
    adv = torch.tensor([1.0], dtype=torch.float64)
    out_default = compute_dppo_binary_policy_loss(log_probs, old, adv, 10.0, 10.0, "dppo_binary_tv")
    out_capped = compute_dppo_binary_policy_loss(log_probs, old, adv, 10.0, 10.0, "dppo_binary_tv", eps_clip_c=2.0)
    lp = math.log(0.9)
    assert torch.allclose(out_default["pg_losses"], torch.tensor([-1.0 * 20.0 * lp], dtype=torch.float64))
    assert torch.allclose(out_capped["pg_losses"], torch.tensor([-1.0 * 2.0 * lp], dtype=torch.float64))


def test_unknown_mode_raises():
    try:
        compute_dppo_binary_policy_loss(_lp([0.5]), _lp([0.5]), torch.tensor([1.0]), 0.2, 0.2, "nope")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
