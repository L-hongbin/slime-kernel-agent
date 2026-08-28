"""R1 support: synthetic SFT-LoRA overfit through the REAL slime launcher.

This module provides the tokenizer-free / parquet-free pieces the real slime
``--debug-train-only --loss-type sft_loss`` launch needs so it can drive the V4 mcore
model + LoRA on a FIXED synthetic batch (same shape as the direct-driver sanity:
B=2 docs, total_len=64, response=32), so the loss curve is comparable.

Pieces:
  * ``v4_sft_rollout``  — a ``--rollout-function-path``: ignores the data source and
    returns a FIXED list of synthetic Samples (tokens / response_length / loss_mask),
    so no tokenizer and no ``--prompt-data`` parquet are needed.  Deterministic per the
    fixed seed so every rollout yields the same batch (overfit-a-batch).
  * ``stage_hf_checkpoint`` — write a tiny ``DeepseekV4Config`` dir + copy a small local
    tokenizer into it, for ``--hf-checkpoint`` (the actor does AutoConfig + AutoTokenizer
    on it; vocab_size comes from the HF config = 512, so the tokenizer's vocab is unused
    by the math — synthetic ids stay < 512).

Use ``--rollout-global-dataset`` OFF + explicit ``--num-rollout N`` so the data source's
``dataset`` is None (it then yields empty ``Sample()`` objects) and ``create_rollout_manager``
skips the epoch calc; ``v4_sft_rollout`` overwrites the empty samples with the fixed batch.
"""

import os
import shutil

# Fixed synthetic batch shape (matches sft_sanity.py for a comparable loss curve).
_TOTAL_LEN = 64
_RESP_LEN = 32
_VOCAB = 512  # must match the tiny DeepseekV4Config vocab_size
_SEED = 123


def _fixed_samples(batch_size: int = 2):
    """Build the fixed list[list[Sample]] (B single-sample groups) with synthetic tokens.

    Deterministic (fixed torch seed) so every rollout returns the IDENTICAL batch — the
    overfit-a-batch signal.  tokens = full [total_len] ids; response = last resp_len;
    loss_mask = all-ones over the response (length == response_length, per the
    _convert_samples_to_train_data assert)."""
    import torch

    from slime.utils.types import Sample

    g = torch.Generator(device="cpu").manual_seed(_SEED)
    groups = []
    for sample_idx in range(batch_size):
        toks = torch.randint(0, _VOCAB, (_TOTAL_LEN,), generator=g).tolist()
        s = Sample()
        s.index = sample_idx
        s.group_id = sample_idx
        s.tokens = toks
        s.response_length = _RESP_LEN
        s.loss_mask = [1] * _RESP_LEN
        s.reward = 0.0
        groups.append([s])  # one sample per group (n_samples_per_prompt handled by slime)
    return groups


def v4_sft_rollout(args, rollout_id, data_source, evaluation=False):
    """slime ``--rollout-function-path``. Returns the fixed synthetic SFT batch.

    Ignores ``data_source`` (no parquet/tokenizer): the fixed batch is the overfit target.
    """
    assert not evaluation, "v4_sft_rollout is train-only (debug-train-only SFT sanity)"
    from slime.rollout.base_types import RolloutFnTrainOutput

    return RolloutFnTrainOutput(
        samples=_fixed_samples(args.rollout_batch_size),
        metrics={"r1_synthetic": 1.0},
    )


def stage_hf_checkpoint(dest: str, tokenizer_src: str | None = None):
    """Write a tiny DeepseekV4Config (vocab=512, the sanity tiny_config) + a tokenizer to
    ``dest`` so ``--hf-checkpoint`` resolves for the actor's AutoConfig + AutoTokenizer.

    tokenizer_src: a local HF dir with a tokenizer (default: a known local Qwen3 dir).
    The tokenizer is only loaded so AutoTokenizer.from_pretrained doesn't crash; the
    synthetic ids (< 512) never go through it.
    """
    from .m0_smoke import tiny_config

    os.makedirs(dest, exist_ok=True)
    cfg = tiny_config()
    cfg.save_pretrained(dest)

    if tokenizer_src is None:
        for cand in (
            "/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3-8B-Base",
            "/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.5-9B",
        ):
            if os.path.exists(os.path.join(cand, "tokenizer.json")):
                tokenizer_src = cand
                break
    assert tokenizer_src and os.path.exists(tokenizer_src), f"no tokenizer src: {tokenizer_src}"
    for fn in os.listdir(tokenizer_src):
        if fn.startswith("tokenizer") or fn in ("special_tokens_map.json", "vocab.json", "merges.txt"):
            shutil.copy(os.path.join(tokenizer_src, fn), os.path.join(dest, fn))
    return dest


__all__ = ["v4_sft_rollout", "stage_hf_checkpoint"]
