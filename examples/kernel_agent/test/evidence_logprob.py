"""Quantitative end-to-end test: per-token DECODE logprob divergence flashinfer vs
triton on the SAME greedy generation. Production mismatch is sglang-decode-logprob
vs megatron; triton-sglang matches megatron; so sglang-fi vs sglang-tri per-token
logprob IS the production-mismatch proxy. Compare on the agreeing-prefix tokens
(identical token id + position = valid teacher-forced comparison). If the per-token
|delta logprob| reaches the production ~0.028 mean / ~9 max, the full-model forward
reproduces the mismatch.

Run per backend (writes JSON), then diff (see inline __main__ diff mode).
"""

import argparse
import json

_PROMPT = (
    "In a distributed reinforcement learning system, the rollout workers generate "
    "token sequences with a fast inference engine while the training backend recomputes "
    "per-token log probabilities for the policy gradient. Write a long detailed essay "
    "about reinforcement learning from human feedback, covering reward models, PPO, GRPO, "
    "KL regularization, and the engineering challenges of large-scale rollout: "
)


def gen(args):
    from sglang import Engine

    eng = Engine(
        model_path=args.model,
        tp_size=1,
        mem_fraction_static=0.85,
        linear_attn_backend=args.backend,
        log_level="warning",
    )
    out = eng.generate(
        _PROMPT, sampling_params={"max_new_tokens": args.max_new, "temperature": 0.0}, return_logprob=True
    )
    eng.shutdown()
    rec = out if isinstance(out, dict) else {}
    meta = rec.get("meta_info", {})
    otl = meta.get("output_token_logprobs", [])  # list of [logprob, token_id, text?]
    ids = [int(x[1]) for x in otl]
    lps = [float(x[0]) for x in otl]
    with open(args.out, "w") as f:
        json.dump({"backend": args.backend, "ids": ids, "logprobs": lps, "text": rec.get("text", "")}, f)
    print(f"[logprob] {args.backend}: {len(ids)} tokens, first ids={ids[:8]}")


def diff(a_path, b_path):
    A = json.load(open(a_path))
    B = json.load(open(b_path))
    ia, la = A["ids"], A["logprobs"]
    ib, lb = B["ids"], B["logprobs"]
    n = min(len(ia), len(ib))
    # compare on the matching prefix (same token id at same position)
    deltas, m = [], 0
    for i in range(n):
        if ia[i] != ib[i]:
            break
        deltas.append(abs(la[i] - lb[i]))
        m += 1
    import statistics

    print(f"agreeing-prefix length = {m} / {n} tokens (first divergent token at pos {m})")
    if deltas:
        print(
            f"per-token |dlogprob| {A['backend']} vs {B['backend']}: "
            f"mean={statistics.mean(deltas):.5f} max={max(deltas):.5f} "
            f"(production: mean=0.028 max=9.22)"
        )
        big = sum(1 for d in deltas if d > 0.001)
        print(f"  tokens with |dlogprob|>0.001 (sequence_mis band): {big}/{m} = {big/m:.1%}")
        # show worst few
        worst = sorted(range(len(deltas)), key=lambda i: -deltas[i])[:6]
        for i in worst:
            print(f"    pos {i}: token={ia[i]} fi_lp={la[i]:.4f} tri_lp={lb[i]:.4f} |d|={deltas[i]:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("flashinfer", "triton"))
    ap.add_argument("--out")
    ap.add_argument("--diff", nargs=2, metavar=("FI", "TRI"))
    ap.add_argument("--model", default="/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B")
    ap.add_argument("--max-new", type=int, default=128)
    a = ap.parse_args()
    if a.diff:
        diff(a.diff[0], a.diff[1])
    else:
        gen(a)
