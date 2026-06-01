#!/usr/bin/env python3
"""Per-turn prompt & answer token-length percentiles for a drkernel eval dump.

For each turn (T1/T2/T3), reports p50/p90/p99 of:
  prompt  = the context entering that turn (tokenized `metadata.turns[t].prompt_snapshot`)
  answer  = that turn's generation incl. thinking (`metadata.turns[t].response_length`)

    python3 per_turn_len.py <path/to/eval_0.pt>

Needs the model tokenizer (re-tokenizes prompt_snapshot). See
handoffs/in_progress/HANDOFF_CACHE_DRKERNEL_HYBRID.md.
"""
import sys

import torch
from transformers import AutoTokenizer

_TOK = "/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B"
tok = AutoTokenizer.from_pretrained(_TOK, trust_remote_code=True)
d = torch.load(sys.argv[1], weights_only=False, map_location="cpu")
S = d["samples"]


def pct(a, p):
    if not a:
        return 0
    a = sorted(a)
    return a[min(len(a) - 1, int(len(a) * p))]


prompt = [[], [], []]
resp = [[], [], []]
for s in S:
    turns = (s.get("metadata") or {}).get("turns") or []
    for t in range(3):
        if t >= len(turns):
            continue
        tr = turns[t]
        ps = tr.get("prompt_snapshot")
        if isinstance(ps, str) and ps:
            prompt[t].append(len(tok.encode(ps, add_special_tokens=False)))
        rl = tr.get("response_length")
        if isinstance(rl, int) and rl > 0:
            resp[t].append(rl)

for name, arrs in (("prompt", prompt), ("answer(incl think)", resp)):
    for t in range(3):
        a = arrs[t]
        print(f"{name:18s} T{t+1}: p50={pct(a,.5):6d}  p90={pct(a,.9):6d}  p99={pct(a,.99):6d}  (n={len(a)})")
