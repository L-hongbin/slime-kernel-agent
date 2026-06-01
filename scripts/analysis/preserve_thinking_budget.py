#!/usr/bin/env python3
"""Bootstrap the 64k context budget for DROP vs PRESERVE thinking, max_turns=3.

See handoffs/in_progress/HANDOFF_CACHE_DRKERNEL_HYBRID.md.
Reads per-turn resp_len from a drkernel eval run.log and estimates peak-context
percentiles + 64k-overflow rates for both regimes.

    python3 preserve_thinking_budget.py [path/to/run.log]

Assumptions (override at top): P0 (first prompt) ~ measured prompt median, feedback
~400 tok, DROP keeps ~20% of each response as answer (reasoning ~80%, reverse-derived
from the measured drop-mode prompt sizes). Bootstrap uses INDEPENDENT sampling of the
three per-turn marginals; real trajectories are likely positively correlated, so the
PRESERVE overflow figure is an UNDER-estimate.
"""
import random
import re
import sys

random.seed(0)
P = (
    sys.argv[1]
    if len(sys.argv) > 1
    else (
        "/nfs/FM/chenshuailin/projects/kernel_agents/slime/checkpoints/Qwen3.6-27B/"
        "20260528_234734_ctx65536_n8_summ1600_nonla_emfrac09_newSlimeKG_tp4/run.log"
    )
)
txt = open(P, encoding="utf-8", errors="ignore").read()


def arr(t):
    return [int(x) for x in re.findall(rf"turn={t}/3 reward=\S+ resp_len=(\d+)", txt)]


R1, R2, R3 = arr(1), arr(2), arr(3)
assert R1 and R2 and R3, "no multi_turn resp_len lines found; wrong run.log?"
P0, FB, CTX, AF, N = 3500, 400, 65536, 0.20, 200000


def pct(a, p):
    a = sorted(a)
    return a[min(len(a) - 1, int(len(a) * p))]


d_blk2 = d_blk3 = d_over = p_blk2 = p_blk3 = p_over = 0
dpk, ppk = [], []
for _ in range(N):
    r1, r2, r3 = random.choice(R1), random.choice(R2), random.choice(R3)
    seq1 = P0 + r1
    # DROP: history keeps only answers (~AF of resp)
    dp2 = P0 + AF * r1 + FB
    dp3 = P0 + AF * r1 + AF * r2 + 2 * FB
    dpeak = max(seq1, dp2 + r2, dp3 + r3)
    dpk.append(dpeak)
    d_blk2 += seq1 > CTX or dp2 >= CTX
    d_blk3 += dp3 >= CTX
    d_over += dpeak > CTX
    # PRESERVE: history keeps full responses
    pp2 = P0 + r1 + FB
    pp3 = P0 + r1 + r2 + 2 * FB
    ppeak = max(seq1, pp2 + r2, pp3 + r3)
    ppk.append(ppeak)
    p_blk2 += seq1 > CTX or pp2 >= CTX
    p_blk3 += pp3 >= CTX
    p_over += ppeak > CTX


def f(x):
    return 100 * x / N


print(
    f"resp1 n={len(R1)} mean={sum(R1)/len(R1):.0f} | resp2 mean={sum(R2)/len(R2):.0f} | resp3 mean={sum(R3)/len(R3):.0f}"
)
print(f"{'metric':42} {'DROP':>10} {'PRESERVE':>10}")
print(f"{'peak-context mean':42} {sum(dpk)/N:10.0f} {sum(ppk)/N:10.0f}")
for q in (0.90, 0.95, 0.99):
    print(f"{'peak-context p'+str(int(q*100)):42} {pct(dpk,q):10d} {pct(ppk,q):10d}")
print(f"{'turn-3 prompt >= 64k (blocked)':42} {f(d_blk3):9.1f}% {f(p_blk3):9.1f}%")
print(f"{'ANY turn overflow 64k':42} {f(d_over):9.1f}% {f(p_over):9.1f}%")
print(f"\n(independent bootstrap N={N}; positive correlation -> PRESERVE overflow is a lower bound)")
