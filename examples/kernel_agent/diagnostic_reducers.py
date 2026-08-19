"""Loss reducers used only by controlled KernelAgent diagnostics.

These are deliberately opt-in through ``--custom-pg-loss-reducer-function-path``.
Production launchers keep slime's normal global-token reduction.
"""

from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean


def get_completion_mean_pg_loss_reducer(
    total_lengths,
    response_lengths,
    loss_masks,
    calculate_per_token_loss,
):
    """Give every completion equal PG weight after its own token mean.

    The outer Megatron loss contract must use the non-per-token path so its
    normalizer is the step sample count.  Accepting the per-token path here
    would divide this already completion-normalized loss by the token count a
    second time and make the A/B invalid, so fail closed.
    """

    if calculate_per_token_loss:
        raise ValueError(
            "completion-mean PG reduction requires CALCULATE_PER_TOKEN_LOSS=0 "
            "so Megatron normalizes by samples rather than tokens"
        )
    return get_sum_of_sample_mean(
        total_lengths,
        response_lengths,
        loss_masks,
        sample_denoms=None,
        calculate_per_token_loss=False,
    )
