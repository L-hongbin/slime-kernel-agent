"""Unit tests for the RLOO advantage path in slime.ray.rollout.group_normalize_rewards.

RLOO leave-one-out advantage: A_i = r_i - mean_{j!=i} r_j = (r_i - group_mean) * g/(g-1).
Implemented as a group-norm branch (center, then * g/(g-1); no std-normalization).
"""

from types import SimpleNamespace

from slime.ray.rollout import group_normalize_rewards


def _run(estimator, rewards, n_samples, batch=1, std_norm=True):
    args = SimpleNamespace(
        advantage_estimator=estimator,
        rewards_normalization=True,
        grpo_std_normalization=std_norm,
        n_samples_per_prompt=n_samples,
        rollout_batch_size=batch,
        reward_key=None,
    )
    raw = [float(r) for r in rewards]
    return raw, group_normalize_rewards(args, raw)


def test_rloo_matches_leave_one_out_definition():
    # one prompt, 4 samples, rewards [1,0,0,0].
    raw, adv = _run("rloo", [1, 0, 0, 0], n_samples=4)
    assert raw == [1.0, 0.0, 0.0, 0.0]
    # A_i = r_i - mean of the OTHER three:
    #   r=1 -> 1 - 0       = 1
    #   r=0 -> 0 - 1/3     = -1/3
    assert abs(adv[0] - 1.0) < 1e-6
    for k in (1, 2, 3):
        assert abs(adv[k] - (-1.0 / 3.0)) < 1e-6
    # leave-one-out baseline is mean-centered => group advantages sum to 0.
    assert abs(sum(adv)) < 1e-6


def test_rloo_all_same_group_collapses_to_zero():
    # No reward variance (all fail / all pass) -> zero advantage (same as grpo).
    _, adv_fail = _run("rloo", [0, 0, 0, 0], n_samples=4)
    _, adv_pass = _run("rloo", [1, 1, 1, 1], n_samples=4)
    assert all(abs(a) < 1e-6 for a in adv_fail)
    assert all(abs(a) < 1e-6 for a in adv_pass)


def test_rloo_uses_g_over_g_minus_1_not_std():
    # rloo scales centered reward by g/(g-1); grpo divides by std. They must differ.
    _, adv_rloo = _run("rloo", [1, 0, 0, 0], n_samples=4)
    _, adv_grpo = _run("grpo", [1, 0, 0, 0], n_samples=4, std_norm=True)
    assert adv_rloo != adv_grpo
    # rloo: centered[0]=0.75, *4/3 = 1.0
    assert abs(adv_rloo[0] - 0.75 * 4 / 3) < 1e-6


def test_rloo_singleton_group_is_zero():
    # g==1 -> no baseline -> 0.
    _, adv = _run("rloo", [1], n_samples=1)
    assert abs(adv[0]) < 1e-6


def test_rloo_two_independent_prompts_grouped_separately():
    # 2 prompts x 2 samples: [1,0 | 0,0]. Each prompt normalized within its own group.
    raw, adv = _run("rloo", [1, 0, 0, 0], n_samples=2, batch=2)
    # prompt 0 [1,0]: g=2 -> A = (r-0.5)*2 -> [1, -1]
    assert abs(adv[0] - 1.0) < 1e-6 and abs(adv[1] - (-1.0)) < 1e-6
    # prompt 1 [0,0]: no variance -> [0, 0]
    assert abs(adv[2]) < 1e-6 and abs(adv[3]) < 1e-6
