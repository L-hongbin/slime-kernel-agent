"""Single-invocation driver for ncu/nsys profiling of one kernel.

Usage: CUDA_VISIBLE_DEVICES=0 python _prof.py {fwd|dq|dkv} [S] [m]
Runs warmup outside the profiled region is not possible under ncu --launch-count,
so we rely on ncu's --launch-skip to skip warmup launches.
"""

import sys
import torch

sys.path.insert(0, ".")
import kernel as K

DEV = "cuda"
H, D = 64, 512

which = sys.argv[1] if len(sys.argv) > 1 else "dkv"
S = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
m = int(sys.argv[3]) if len(sys.argv) > 3 else 4
W = 128
B = 1
block_M = block_N = 64
Tcomp = S // m

q = torch.randn(B, H, S, D, device=DEV, dtype=torch.bfloat16) * 0.3
kr = torch.randn(B, 1, S, D, device=DEV, dtype=torch.bfloat16) * 0.3
kc = torch.randn(B, 1, Tcomp, D, device=DEV, dtype=torch.bfloat16) * 0.3
sinks = torch.randn(H, device=DEV, dtype=torch.float32)

mb = max(block_M, block_N)
Sp = K._ceil(S, mb) * mb
Tcp = K._ceil(Tcomp, block_N) * block_N
KV = Sp + Tcp
q_pad = q.new_zeros(B, H, Sp, D)
q_pad[:, :, :S] = q
kvt = q.new_zeros(B, 1, KV, D)
kvt[:, :, :S] = kr
kvt[:, :, Sp : Sp + Tcomp] = kc
sinks_f = sinks.float().contiguous()

args = (B, H, S, Tcomp, Sp, Tcp, W, m, D, block_M, block_N)
key = args + (str(torch.bfloat16),)
fwd = K._get(K._FWD_CACHE, K._build_fwd, key, *args)
out_pad, lse_pad = fwd(q_pad, kvt, sinks_f)
dO = q_pad.new_zeros(B, H, Sp, D)
dO[:, :, :S] = torch.randn(B, H, S, D, device=DEV, dtype=torch.bfloat16)
delta = (dO.float() * out_pad.float()).sum(-1).contiguous()
dKV = torch.zeros(B, 1, KV, D, device=DEV, dtype=torch.float32)

if which == "fwd":
    fn = lambda: fwd(q_pad, kvt, sinks_f)
elif which == "dq":
    dq_fn = K._get(K._DQ_CACHE, K._build_bwd_dq, args, *args)
    fn = lambda: dq_fn(q_pad, kvt, dO, lse_pad, delta)
else:
    dkv_raw = K._get(K._DKV_RAW_CACHE, K._build_bwd_dkv_raw, args, *args)
    dkv_comp = K._get(K._DKV_COMP_CACHE, K._build_bwd_dkv_comp, args, *args)

    def fn():
        dKV.zero_()
        dkv_raw(q_pad, kvt, dO, lse_pad, delta, dKV)
        dkv_comp(q_pad, kvt, dO, lse_pad, delta, dKV)


# warmup launches (ncu --launch-skip will skip these)
for _ in range(5):
    fn()
torch.cuda.synchronize()
for _ in range(3):
    fn()
torch.cuda.synchronize()
print("done", which, S, m)
