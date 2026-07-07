"""DeepSeek-V4-Flash core attention -- tilelang forward + hand-written backward.

Implements the A1 kernel from `handoffs/deepseek-v4/v4_kernel_inventory.md`:
shared-KV MQA (K == V, one KV head broadcast to all H heads), head_dim=512,
per-head attention sink (gpt-oss style, denom-only), and an additive structural
mask over a KV axis of `[ raw (length S, sliding-window causal) ++ compressed
(length Tcomp, causal-threshold) ]`.  bf16 compute, fp32 softmax/accumulation.

The forward is an online-softmax flash kernel that loads each KV tile ONCE and
reuses it for both `Q.Kᵀ` and `P.V` (legal because K == V here).  Raw and
compressed KV live in one concatenated tensor `[raw_pad (Sp) ++ comp (Tcp)]`
and are walked by a SINGLE kv loop so the `acc_o` accumulator keeps one
consistent fragment layout (two separate loops gave `acc_o` inconsistent
layouts across loop scopes -> wrong numerator).

Block-sparsity is exploited structurally: the loop only visits the banded
sliding-window raw blocks plus the compressed blocks below the per-block causal
threshold; fully-masked blocks are never iterated.

The backward is hand-written (no autodiff); see `_build_bwd_*`.

See `reference.py` for the exact math and `test_correctness.py` for validation.
"""

import os

import tilelang
import tilelang.language as T
import torch

NEG = -1e30  # finite stand-in for -inf masked logits (avoids inf-inf NaNs)


def _ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


# --------------------------------------------------------------------------- #
# Forward kernel                                                              #
# --------------------------------------------------------------------------- #
def _build_fwd(B, H, S, Tcomp, Sp, Tcp, window, m, D, block_M=64, block_N=64, threads=128, stages=1, dtype="bfloat16"):
    scaling = float(D**-0.5)
    accum = "float32"
    n_q = Sp // block_M
    KV = Sp + Tcp  # concatenated raw_pad ++ comp
    raw_blocks = Sp // block_N  # number of raw kv blocks (Sp % block_N == 0)

    @T.prim_func
    def kernel(
        Q: T.Tensor([B, H, Sp, D], dtype),
        KVt: T.Tensor([B, 1, KV, D], dtype),  # [raw_pad (Sp) ++ comp (Tcp)]
        Sinks: T.Tensor([H], accum),
        Out: T.Tensor([B, H, Sp, D], dtype),
        Lse: T.Tensor([B, H, Sp], accum),
    ):
        with T.Kernel(n_q, H, B, threads=threads) as (bx, by, bz):
            Q_sh = T.alloc_shared([block_M, D], dtype)
            K_sh = T.alloc_shared([block_N, D], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum)
            acc_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, D], accum)
            m_i = T.alloc_fragment([block_M], accum)
            m_prev = T.alloc_fragment([block_M], accum)
            m_cur = T.alloc_fragment([block_M], accum)
            scale_f = T.alloc_fragment([block_M], accum)
            l_i = T.alloc_fragment([block_M], accum)
            row_sum = T.alloc_fragment([block_M], accum)

            q0 = bx * block_M
            T.copy(Q[bz, by, q0 : q0 + block_M, :], Q_sh)
            T.fill(acc_o, 0.0)
            T.fill(l_i, 0.0)
            T.fill(m_i, NEG)

            # banded raw block range [raw_lo, raw_hi] (sliding window)
            raw_lo = T.max(0, (q0 - (window - 1)) // block_N)
            raw_hi = (q0 + block_M - 1) // block_N  # inclusive
            n_raw = raw_hi + 1 - raw_lo
            # compressed block count below the block's causal threshold
            if Tcomp > 0:
                comp_blocks = T.ceildiv(T.min((q0 + block_M) // m, Tcomp), block_N)
            else:
                comp_blocks = 0
            n_total = n_raw + comp_blocks

            for t in T.Pipelined(0, n_total, num_stages=stages):
                is_raw = t < n_raw
                blk = T.if_then_else(is_raw, raw_lo + t, raw_blocks + (t - n_raw))
                base = blk * block_N
                T.copy(KVt[bz, 0, base : base + block_N, :], K_sh)
                T.clear(acc_s)
                T.gemm(Q_sh, K_sh, acc_s, transpose_B=True)
                for i, j in T.Parallel(block_M, block_N):
                    # Inlined mask (no named bool sub-exprs -> avoids a TVM
                    # analyzer crash where an unrolled Let var gets conflicting
                    # const bounds). raw: sliding-window causal; comp: causal
                    # threshold (w+1)*m <= qpos+1  <=>  w < (qpos+1)//m.
                    acc_s[i, j] = T.if_then_else(
                        T.if_then_else(
                            t < n_raw,
                            (base + j < S) and (base + j <= q0 + i) and (q0 + i - (base + j) < window),
                            (base + j - Sp < Tcomp) and ((base + j - Sp + 1) * m <= q0 + i + 1),
                        ),
                        acc_s[i, j] * scaling,
                        NEG,
                    )
                # --- online-softmax merge (K_sh doubles as V, K == V) ---
                T.copy(m_i, m_prev)
                T.reduce_max(acc_s, m_cur, dim=1, clear=True)
                for i in T.Parallel(block_M):
                    m_i[i] = T.max(m_prev[i], m_cur[i])
                for i in T.Parallel(block_M):
                    scale_f[i] = T.if_then_else(m_i[i] <= NEG, 0.0, T.exp(m_prev[i] - m_i[i]))
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(m_i[i] <= NEG, 0.0, T.exp(acc_s[i, j] - m_i[i]))
                T.reduce_sum(acc_s, row_sum, dim=1, clear=True)
                for i in T.Parallel(block_M):
                    l_i[i] = l_i[i] * scale_f[i] + row_sum[i]
                for i, j in T.Parallel(block_M, D):
                    acc_o[i, j] = acc_o[i, j] * scale_f[i]
                T.copy(acc_s, acc_cast)
                T.gemm(acc_cast, K_sh, acc_o)

            # ---- sink + normalize ----
            for i in T.Parallel(block_M):
                sink_v = Sinks[by]
                m_new = T.max(m_i[i], sink_v)
                corr = T.if_then_else(m_i[i] <= NEG, 0.0, T.exp(m_i[i] - m_new))
                l_new = l_i[i] * corr + T.exp(sink_v - m_new)
                scale_f[i] = corr / l_new
                Lse[bz, by, q0 + i] = m_new + T.log(l_new)
            for i, j in T.Parallel(block_M, D):
                acc_o[i, j] = acc_o[i, j] * scale_f[i]
            T.copy(acc_o, Out[bz, by, q0 : q0 + block_M, :])

    return tilelang.compile(kernel, out_idx=[-2, -1])


# --------------------------------------------------------------------------- #
# Warp-specialized split-D forward (2 consumer warpgroups, head_dim halved):  #
# WG0 = producer+consumer-half0 (loads K/V, QK, softmax, publishes probs +     #
# scale to shared, PV over D[0:Dh]); WG1 = consumer-half1 (PV over D[Dh:D]).    #
# Cross-warpgroup ordering via NAMED barriers (bar.sync id,256) hit in lockstep #
# by both warpgroups -> deadlock-safe. TMA/auto-warpspec disabled so manual ws  #
# is the only specialization.                                                   #
# --------------------------------------------------------------------------- #
def _build_fwd_ws(
    B, H, S, Tcomp, Sp, Tcp, window, m, D, block_M=64, block_N=64, threads=256, stages=1, dtype="bfloat16"
):
    scaling = float(D**-0.5)
    accum = "float32"
    n_q = Sp // block_M
    KV = Sp + Tcp
    raw_blocks = Sp // block_N
    Dh = D // 2
    BA, BB, BC, BD = 1, 2, 3, 4  # named-barrier ids

    @T.macro
    def _wg0(Q, KVt, Sinks, Out, Lse, Q0_sh, Q1_sh, V0_sh, V1_sh, P_sh, sc_sh, fsc_sh, bx, by, bz):
        acc_s = T.alloc_fragment([block_M, block_N], accum)
        acc_cast = T.alloc_fragment([block_M, block_N], dtype)
        acc_o = T.alloc_fragment([block_M, Dh], accum)
        m_i = T.alloc_fragment([block_M], accum)
        m_prev = T.alloc_fragment([block_M], accum)
        m_cur = T.alloc_fragment([block_M], accum)
        scale_f = T.alloc_fragment([block_M], accum)
        l_i = T.alloc_fragment([block_M], accum)
        row_sum = T.alloc_fragment([block_M], accum)

        q0 = bx * block_M
        T.copy(Q[bz, by, q0 : q0 + block_M, 0:Dh], Q0_sh)
        T.copy(Q[bz, by, q0 : q0 + block_M, Dh:D], Q1_sh)
        T.fill(acc_o, 0.0)
        T.fill(l_i, 0.0)
        T.fill(m_i, NEG)

        raw_lo = T.max(0, (q0 - (window - 1)) // block_N)
        raw_hi = (q0 + block_M - 1) // block_N
        n_raw = raw_hi + 1 - raw_lo
        if Tcomp > 0:
            comp_blocks = T.ceildiv(T.min((q0 + block_M) // m, Tcomp), block_N)
        else:
            comp_blocks = 0
        n_total = n_raw + comp_blocks

        for t in T.serial(0, n_total):
            is_raw = t < n_raw
            blk = T.if_then_else(is_raw, raw_lo + t, raw_blocks + (t - n_raw))
            base = blk * block_N
            T.copy(KVt[bz, 0, base : base + block_N, 0:Dh], V0_sh)
            T.copy(KVt[bz, 0, base : base + block_N, Dh:D], V1_sh)
            T.clear(acc_s)
            T.gemm(Q0_sh, V0_sh, acc_s, transpose_B=True)
            T.gemm(Q1_sh, V1_sh, acc_s, transpose_B=True)  # full QK = sum of halves
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.if_then_else(
                    T.if_then_else(
                        t < n_raw,
                        (base + j < S) and (base + j <= q0 + i) and (q0 + i - (base + j) < window),
                        (base + j - Sp < Tcomp) and ((base + j - Sp + 1) * m <= q0 + i + 1),
                    ),
                    acc_s[i, j] * scaling,
                    NEG,
                )
            T.copy(m_i, m_prev)
            T.reduce_max(acc_s, m_cur, dim=1, clear=True)
            for i in T.Parallel(block_M):
                m_i[i] = T.max(m_prev[i], m_cur[i])
            for i in T.Parallel(block_M):
                scale_f[i] = T.if_then_else(m_i[i] <= NEG, 0.0, T.exp(m_prev[i] - m_i[i]))
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.if_then_else(m_i[i] <= NEG, 0.0, T.exp(acc_s[i, j] - m_i[i]))
            T.reduce_sum(acc_s, row_sum, dim=1, clear=True)
            for i in T.Parallel(block_M):
                l_i[i] = l_i[i] * scale_f[i] + row_sum[i]
            T.copy(acc_s, acc_cast)
            T.copy(acc_cast, P_sh)  # publish probs to WG1
            T.copy(scale_f, sc_sh)  # publish per-iter rescale to WG1
            T.sync_threads(BB, 256)  # probs + scale ready for WG1
            for i, j in T.Parallel(block_M, Dh):
                acc_o[i, j] = acc_o[i, j] * scale_f[i]
            T.gemm(acc_cast, V0_sh, acc_o)
            T.sync_threads(BC, 256)  # both consumers done -> safe to reload V

        for i in T.Parallel(block_M):
            sink_v = Sinks[by]
            m_new = T.max(m_i[i], sink_v)
            corr = T.if_then_else(m_i[i] <= NEG, 0.0, T.exp(m_i[i] - m_new))
            l_new = l_i[i] * corr + T.exp(sink_v - m_new)
            scale_f[i] = corr / l_new
            Lse[bz, by, q0 + i] = m_new + T.log(l_new)
        T.copy(scale_f, fsc_sh)  # publish FINAL scale to WG1
        T.sync_threads(BD, 256)
        for i, j in T.Parallel(block_M, Dh):
            acc_o[i, j] = acc_o[i, j] * scale_f[i]
        T.copy(acc_o, Out[bz, by, q0 : q0 + block_M, 0:Dh])

    @T.macro
    def _wg1(Q, KVt, Out, P_sh, sc_sh, fsc_sh, V1_sh, bx, by, bz):
        acc_cast = T.alloc_fragment([block_M, block_N], dtype)
        acc_o = T.alloc_fragment([block_M, Dh], accum)
        sc = T.alloc_fragment([block_M], accum)
        fsc = T.alloc_fragment([block_M], accum)

        q0 = bx * block_M
        T.fill(acc_o, 0.0)
        raw_lo = T.max(0, (q0 - (window - 1)) // block_N)
        raw_hi = (q0 + block_M - 1) // block_N
        n_raw = raw_hi + 1 - raw_lo
        if Tcomp > 0:
            comp_blocks = T.ceildiv(T.min((q0 + block_M) // m, Tcomp), block_N)
        else:
            comp_blocks = 0
        n_total = n_raw + comp_blocks

        for t in T.serial(0, n_total):
            T.sync_threads(BB, 256)  # wait probs + scale ready
            T.copy(P_sh, acc_cast)
            T.copy(sc_sh, sc)
            for i, j in T.Parallel(block_M, Dh):
                acc_o[i, j] = acc_o[i, j] * sc[i]
            T.gemm(acc_cast, V1_sh, acc_o)
            T.sync_threads(BC, 256)  # signal consumer done

        T.sync_threads(BD, 256)  # wait final scale
        T.copy(fsc_sh, fsc)
        for i, j in T.Parallel(block_M, Dh):
            acc_o[i, j] = acc_o[i, j] * fsc[i]
        T.copy(acc_o, Out[bz, by, q0 : q0 + block_M, Dh:D])

    @T.prim_func
    def kernel(
        Q: T.Tensor([B, H, Sp, D], dtype),
        KVt: T.Tensor([B, 1, KV, D], dtype),
        Sinks: T.Tensor([H], accum),
        Out: T.Tensor([B, H, Sp, D], dtype),
        Lse: T.Tensor([B, H, Sp], accum),
    ):
        with T.Kernel(n_q, H, B, threads=threads) as (bx, by, bz):
            Q0_sh = T.alloc_shared([block_M, Dh], dtype)
            Q1_sh = T.alloc_shared([block_M, Dh], dtype)
            V0_sh = T.alloc_shared([block_N, Dh], dtype)
            V1_sh = T.alloc_shared([block_N, Dh], dtype)
            P_sh = T.alloc_shared([block_M, block_N], dtype)
            sc_sh = T.alloc_shared([block_M], accum)
            fsc_sh = T.alloc_shared([block_M], accum)
            with T.ws(0):
                _wg0(Q, KVt, Sinks, Out, Lse, Q0_sh, Q1_sh, V0_sh, V1_sh, P_sh, sc_sh, fsc_sh, bx, by, bz)
            with T.ws(1):
                _wg1(Q, KVt, Out, P_sh, sc_sh, fsc_sh, V1_sh, bx, by, bz)

    return tilelang.compile(
        kernel, out_idx=[-2, -1], pass_configs={"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True}
    )


# --------------------------------------------------------------------------- #
# Backward: dQ kernel                                                         #
# --------------------------------------------------------------------------- #
# Recompute path (no autodiff). Stored from fwd: lse_i (incl. sink) and
# delta_i = sum_d dO[i,d] * O[i,d].  For valid (i,j):
#   p_ij  = exp(scaling * q_i.k_j - lse_i)          (0 where masked)
#   dp_ij = dO_i . v_j           (v_j == kv_j)
#   dS_ij = p_ij * (dp_ij - delta_i)
#   dQ_i  = scaling * sum_j dS_ij k_j
# Single unified kv loop (raw_pad ++ comp), same banding as fwd.
def _build_bwd_dq(
    B, H, S, Tcomp, Sp, Tcp, window, m, D, block_M=64, block_N=64, threads=128, stages=1, dtype="bfloat16"
):
    scaling = float(D**-0.5)
    accum = "float32"
    n_q = Sp // block_M
    KV = Sp + Tcp
    raw_blocks = Sp // block_N

    @T.prim_func
    def kernel(
        Q: T.Tensor([B, H, Sp, D], dtype),
        KVt: T.Tensor([B, 1, KV, D], dtype),
        dO: T.Tensor([B, H, Sp, D], dtype),
        Lse: T.Tensor([B, H, Sp], accum),
        Delta: T.Tensor([B, H, Sp], accum),
        dQ: T.Tensor([B, H, Sp, D], accum),
    ):
        with T.Kernel(n_q, H, B, threads=threads) as (bx, by, bz):
            Q_sh = T.alloc_shared([block_M, D], dtype)
            K_sh = T.alloc_shared([block_N, D], dtype)
            dO_sh = T.alloc_shared([block_M, D], dtype)
            s = T.alloc_fragment([block_M, block_N], accum)
            dp = T.alloc_fragment([block_M, block_N], accum)
            dS_cast = T.alloc_fragment([block_M, block_N], dtype)
            dq_acc = T.alloc_fragment([block_M, D], accum)
            lse_f = T.alloc_fragment([block_M], accum)
            delta_f = T.alloc_fragment([block_M], accum)

            q0 = bx * block_M
            T.copy(Q[bz, by, q0 : q0 + block_M, :], Q_sh)
            T.copy(dO[bz, by, q0 : q0 + block_M, :], dO_sh)
            T.copy(Lse[bz, by, q0 : q0 + block_M], lse_f)
            T.copy(Delta[bz, by, q0 : q0 + block_M], delta_f)
            T.fill(dq_acc, 0.0)

            raw_lo = T.max(0, (q0 - (window - 1)) // block_N)
            raw_hi = (q0 + block_M - 1) // block_N
            n_raw = raw_hi + 1 - raw_lo
            if Tcomp > 0:
                comp_blocks = T.ceildiv(T.min((q0 + block_M) // m, Tcomp), block_N)
            else:
                comp_blocks = 0
            n_total = n_raw + comp_blocks

            for t in T.Pipelined(0, n_total, num_stages=stages):
                is_raw = t < n_raw
                blk = T.if_then_else(is_raw, raw_lo + t, raw_blocks + (t - n_raw))
                base = blk * block_N
                T.copy(KVt[bz, 0, base : base + block_N, :], K_sh)
                T.clear(s)
                T.gemm(Q_sh, K_sh, s, transpose_B=True)  # s_ij = q_i.k_j
                T.clear(dp)
                T.gemm(dO_sh, K_sh, dp, transpose_B=True)  # dp_ij = dO_i.k_j
                for i, j in T.Parallel(block_M, block_N):
                    p = T.if_then_else(
                        T.if_then_else(
                            t < n_raw,
                            (base + j < S) and (base + j <= q0 + i) and (q0 + i - (base + j) < window),
                            (base + j - Sp < Tcomp) and ((base + j - Sp + 1) * m <= q0 + i + 1),
                        ),
                        T.exp(scaling * s[i, j] - lse_f[i]),
                        0.0,
                    )
                    dS_cast[i, j] = scaling * (p * (dp[i, j] - delta_f[i]))
                T.gemm(dS_cast, K_sh, dq_acc)  # dQ += scaling*dS @ K
            T.copy(dq_acc, dQ[bz, by, q0 : q0 + block_M, :])

    return tilelang.compile(kernel, out_idx=[-1])


# --------------------------------------------------------------------------- #
# Backward: dKV kernels (shared-KV: reduce over all H heads via atomic add)   #
# --------------------------------------------------------------------------- #
# For a kv block (keys j), loop the query blocks that attend it:
#   dv_j = sum_i p_ij  dO_i
#   dk_j = scaling * sum_i dS_ij q_i
# d(kv_j) = dk_j + dv_j accumulated into one [BN,D] tile, then atomic-added into
# dKV[b,0,j,:] (the atomic sums the broadcast contribution over all H heads).
#
# Split into TWO kernels (raw vs compressed) so each gets a TIGHT query-block
# range with no `is_raw` branch in the inner mask:
#   * raw block kv0 in [0, Sp): banded sliding-window  q in [kv0/BM, ...].
#   * compressed block: entry w = kv0c + j is attended only by queries
#     i >= (w+1)*m - 1, so the tight lower bound for the whole block (min w =
#     kv0c) is q_lo = ((kv0c+1)*m - 1) // BM -- instead of scanning from q=0.
#     The per-element mask stays (partial diagonal tiles), so the analyzer
#     cannot fold `valid` to a constant -> no lowering crash.
def _build_bwd_dkv_raw(
    B, H, S, Tcomp, Sp, Tcp, window, m, D, block_M=64, block_N=64, threads=128, stages=1, dtype="bfloat16"
):
    scaling = float(D**-0.5)
    accum = "float32"
    n_q = Sp // block_M
    KV = Sp + Tcp
    raw_blocks = Sp // block_N

    @T.prim_func
    def kernel(
        Q: T.Tensor([B, H, Sp, D], dtype),
        KVt: T.Tensor([B, 1, KV, D], dtype),
        dO: T.Tensor([B, H, Sp, D], dtype),
        Lse: T.Tensor([B, H, Sp], accum),
        Delta: T.Tensor([B, H, Sp], accum),
        dKV: T.Tensor([B, 1, KV, D], accum),
    ):
        with T.Kernel(raw_blocks, H, B, threads=threads) as (bx, by, bz):
            Kj_sh = T.alloc_shared([block_N, D], dtype)
            Q_sh = T.alloc_shared([block_M, D], dtype)
            dO_sh = T.alloc_shared([block_M, D], dtype)
            s = T.alloc_fragment([block_N, block_M], accum)
            dp = T.alloc_fragment([block_N, block_M], accum)
            p_cast = T.alloc_fragment([block_N, block_M], dtype)
            dS_cast = T.alloc_fragment([block_N, block_M], dtype)
            dkv_acc = T.alloc_fragment([block_N, D], accum)
            lse_f = T.alloc_fragment([block_M], accum)
            delta_f = T.alloc_fragment([block_M], accum)

            kv0 = bx * block_N
            T.copy(KVt[bz, 0, kv0 : kv0 + block_N, :], Kj_sh)
            T.fill(dkv_acc, 0.0)

            # banded sliding-window query range for this raw kv block
            q_lo = kv0 // block_M
            q_hi = T.min(n_q - 1, (kv0 + block_N - 1 + window - 1) // block_M)

            for tq in T.Pipelined(q_lo, q_hi + 1, num_stages=stages):
                q0 = tq * block_M
                T.copy(Q[bz, by, q0 : q0 + block_M, :], Q_sh)
                T.copy(dO[bz, by, q0 : q0 + block_M, :], dO_sh)
                T.copy(Lse[bz, by, q0 : q0 + block_M], lse_f)
                T.copy(Delta[bz, by, q0 : q0 + block_M], delta_f)
                T.clear(s)
                T.gemm(Kj_sh, Q_sh, s, transpose_B=True)  # s_ji = k_j.q_i
                T.clear(dp)
                T.gemm(Kj_sh, dO_sh, dp, transpose_B=True)  # dp_ji = k_j.dO_i
                for j, i in T.Parallel(block_N, block_M):
                    p = T.if_then_else(
                        (kv0 + j < S) and (kv0 + j <= q0 + i) and (q0 + i - (kv0 + j) < window),
                        T.exp(scaling * s[j, i] - lse_f[i]),
                        0.0,
                    )
                    p_cast[j, i] = p
                    dS_cast[j, i] = scaling * (p * (dp[j, i] - delta_f[i]))
                T.gemm(dS_cast, Q_sh, dkv_acc)  # dk: += scaling*dS @ Q
                T.gemm(p_cast, dO_sh, dkv_acc)  # dv: += p @ dO
            T.atomic_add(dKV[bz, 0, kv0 : kv0 + block_N, :], dkv_acc)

    return tilelang.compile(kernel)


def _build_bwd_dkv_comp(
    B, H, S, Tcomp, Sp, Tcp, window, m, D, block_M=64, block_N=64, threads=128, stages=1, dtype="bfloat16"
):
    scaling = float(D**-0.5)
    accum = "float32"
    n_q = Sp // block_M
    KV = Sp + Tcp
    comp_blocks = Tcp // block_N

    @T.prim_func
    def kernel(
        Q: T.Tensor([B, H, Sp, D], dtype),
        KVt: T.Tensor([B, 1, KV, D], dtype),
        dO: T.Tensor([B, H, Sp, D], dtype),
        Lse: T.Tensor([B, H, Sp], accum),
        Delta: T.Tensor([B, H, Sp], accum),
        dKV: T.Tensor([B, 1, KV, D], accum),
    ):
        with T.Kernel(comp_blocks, H, B, threads=threads) as (bx, by, bz):
            Kj_sh = T.alloc_shared([block_N, D], dtype)
            Q_sh = T.alloc_shared([block_M, D], dtype)
            dO_sh = T.alloc_shared([block_M, D], dtype)
            s = T.alloc_fragment([block_N, block_M], accum)
            dp = T.alloc_fragment([block_N, block_M], accum)
            p_cast = T.alloc_fragment([block_N, block_M], dtype)
            dS_cast = T.alloc_fragment([block_N, block_M], dtype)
            dkv_acc = T.alloc_fragment([block_N, D], accum)
            lse_f = T.alloc_fragment([block_M], accum)
            delta_f = T.alloc_fragment([block_M], accum)

            wc0 = bx * block_N  # compressed entry index of this block
            kv0 = Sp + wc0  # offset into the concatenated KV buffer
            T.copy(KVt[bz, 0, kv0 : kv0 + block_N, :], Kj_sh)
            T.fill(dkv_acc, 0.0)

            # TIGHT lower query bound: min entry in block is wc0, attended only by
            # queries i >= (wc0+1)*m - 1.  All later queries attend (more entries
            # valid as i grows), so q_hi = n_q-1.  Clamp to n_q so a fully-padded
            # compressed block (wc0 >= Tcomp, only possible when Tcomp % block_N
            # != 0) yields a zero-extent loop instead of a negative one.
            q_lo = T.min(n_q, T.max(0, ((wc0 + 1) * m - 1) // block_M))

            for tq in T.Pipelined(q_lo, n_q, num_stages=stages):
                q0 = tq * block_M
                T.copy(Q[bz, by, q0 : q0 + block_M, :], Q_sh)
                T.copy(dO[bz, by, q0 : q0 + block_M, :], dO_sh)
                T.copy(Lse[bz, by, q0 : q0 + block_M], lse_f)
                T.copy(Delta[bz, by, q0 : q0 + block_M], delta_f)
                T.clear(s)
                T.gemm(Kj_sh, Q_sh, s, transpose_B=True)  # s_ji = k_j.q_i
                T.clear(dp)
                T.gemm(Kj_sh, dO_sh, dp, transpose_B=True)  # dp_ji = k_j.dO_i
                for j, i in T.Parallel(block_N, block_M):
                    p = T.if_then_else(
                        (wc0 + j < Tcomp) and ((wc0 + j + 1) * m <= q0 + i + 1),
                        T.exp(scaling * s[j, i] - lse_f[i]),
                        0.0,
                    )
                    p_cast[j, i] = p
                    dS_cast[j, i] = scaling * (p * (dp[j, i] - delta_f[i]))
                T.gemm(dS_cast, Q_sh, dkv_acc)  # dk: += scaling*dS @ Q
                T.gemm(p_cast, dO_sh, dkv_acc)  # dv: += p @ dO
            T.atomic_add(dKV[bz, 0, kv0 : kv0 + block_N, :], dkv_acc)

    return tilelang.compile(kernel)


# --------------------------------------------------------------------------- #
# autograd.Function wrapper + public API                                      #
# --------------------------------------------------------------------------- #
# Warp-specialized split-D forward (default). V4_ATTN_WS_FWD=0 falls back to the
# single-warpgroup baseline forward (byte-identical Out/Lse; ~1.3x slower) —
# same escape-hatch pattern as V4_*_TORCH for nodes with TileLang codegen issues.
_USE_WS_FWD = os.environ.get("V4_ATTN_WS_FWD", "1") == "1"
_FWD_CACHE: dict = {}
_DQ_CACHE: dict = {}
_DKV_RAW_CACHE: dict = {}
_DKV_COMP_CACHE: dict = {}


def _get(cache, builder, key, *args):
    k = cache.get(key)
    if k is None:
        k = builder(*args)
        cache[key] = k
    return k


class _V4FlashAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k_raw, k_comp, sinks, window, m, block_M, block_N):
        # q [B,H,S,D]; k_raw [B,1,S,D]; k_comp [B,1,Tcomp,D] or None; sinks [H]
        B, H, S, D = q.shape
        Tcomp = 0 if k_comp is None else k_comp.shape[2]
        Sp = _ceil(S, max(block_M, block_N)) * max(block_M, block_N)
        Tcp = _ceil(Tcomp, block_N) * block_N if Tcomp > 0 else 0
        KV = Sp + Tcp
        dev, dt = q.device, q.dtype

        q_pad = q.new_zeros(B, H, Sp, D)
        q_pad[:, :, :S] = q
        kvt = q.new_zeros(B, 1, KV, D)
        kvt[:, :, :S] = k_raw
        if Tcomp > 0:
            kvt[:, :, Sp : Sp + Tcomp] = k_comp
        sinks_f = sinks.float().contiguous()

        # Warp-specialized split-D forward (2 consumer warpgroups; ~1.3x dense,
        # ~1.1x sparse vs the single-warpgroup _build_fwd). Produces byte-identical
        # Lse/Out so the hand-written backward is unchanged. Set V4_ATTN_WS_FWD=0
        # to fall back to the single-warpgroup baseline.
        fwd_builder = _build_fwd_ws if _USE_WS_FWD else _build_fwd
        key = (B, H, S, Tcomp, Sp, Tcp, window, m, D, block_M, block_N, str(dt), _USE_WS_FWD)
        fwd = _get(_FWD_CACHE, fwd_builder, key, B, H, S, Tcomp, Sp, Tcp, window, m, D, block_M, block_N)
        out_pad, lse_pad = fwd(q_pad, kvt, sinks_f)

        ctx.save_for_backward(q_pad, kvt, sinks_f, out_pad, lse_pad)
        ctx.dims = (B, H, S, Tcomp, Sp, Tcp, window, m, D, block_M, block_N)
        ctx.has_comp = Tcomp > 0
        ctx.q_dtype, ctx.kr_dtype = q.dtype, k_raw.dtype
        ctx.kc_dtype = None if k_comp is None else k_comp.dtype
        return out_pad[:, :, :S]

    @staticmethod
    def backward(ctx, dout):
        q_pad, kvt, sinks_f, out_pad, lse_pad = ctx.saved_tensors
        (B, H, S, Tcomp, Sp, Tcp, window, m, D, block_M, block_N) = ctx.dims
        KV = Sp + Tcp

        dO = q_pad.new_zeros(B, H, Sp, D)
        dO[:, :, :S] = dout
        # delta_i = sum_d dO_i . O_i   (fp32)
        delta = (dO.float() * out_pad.float()).sum(-1).contiguous()  # [B,H,Sp]

        kdq = (B, H, S, Tcomp, Sp, Tcp, window, m, D, block_M, block_N)
        dq_fn = _get(_DQ_CACHE, _build_bwd_dq, kdq, B, H, S, Tcomp, Sp, Tcp, window, m, D, block_M, block_N)
        dkv_raw_fn = _get(
            _DKV_RAW_CACHE, _build_bwd_dkv_raw, kdq, B, H, S, Tcomp, Sp, Tcp, window, m, D, block_M, block_N
        )

        dQ = dq_fn(q_pad, kvt, dO, lse_pad, delta)  # [B,H,Sp,D] fp32
        dKV = torch.zeros(B, 1, KV, D, device=q_pad.device, dtype=torch.float32)
        dkv_raw_fn(q_pad, kvt, dO, lse_pad, delta, dKV)  # raw, atomic-accumulated
        if Tcp > 0:
            dkv_comp_fn = _get(
                _DKV_COMP_CACHE, _build_bwd_dkv_comp, kdq, B, H, S, Tcomp, Sp, Tcp, window, m, D, block_M, block_N
            )
            dkv_comp_fn(q_pad, kvt, dO, lse_pad, delta, dKV)  # compressed, atomic-accumulated

        # dsink_h = -sum_{b,i<S} exp(sink_h - lse_i) * delta_i.  ``sinks`` is frozen under
        # LoRA, so skip the param-grad compute when autograd says it isn't needed
        # (ctx.needs_input_grad[3] == sinks); only the input grads dq/dk feed cross-layer
        # flow.  When sinks IS trained (full FT) the grad is computed as before.
        if ctx.needs_input_grad[3]:
            lse_r = lse_pad[:, :, :S].float()
            delta_r = delta[:, :, :S]
            p_sink = torch.exp(sinks_f.view(1, H, 1) - lse_r)  # [B,H,S]
            dsink = -(p_sink * delta_r).sum(dim=(0, 2)).to(sinks_f.dtype)
        else:
            dsink = None

        dq = dQ[:, :, :S].to(ctx.q_dtype)
        dk_raw = dKV[:, :, :S].to(ctx.kr_dtype)
        dk_comp = None
        if ctx.has_comp:
            dk_comp = dKV[:, :, Sp : Sp + Tcomp].to(ctx.kc_dtype)
        return dq, dk_raw, dk_comp, dsink, None, None, None, None


def v4flash_attention(q, k_raw, k_comp, sinks, window, m, block_M=64, block_N=64):
    """DeepSeek-V4-Flash core attention (training; no KV cache).

    Args:
        q:      [B, H, S, D]   post-RoPE queries.
        k_raw:  [B, 1, S, D]   post-RoPE raw KV (K == V).
        k_comp: [B, 1, Tcomp, D] or None   post-RoPE compressed KV (K == V).
        sinks:  [H]            per-head sink logit (frozen base param).
        window: sliding window W.
        m:      compress rate (CSA 4 / HCA 128); ignored if k_comp is None.

    Returns:
        out: [B, H, S, D]   attention output (conjugate-RoPE / o_proj are external).
    """
    return _V4FlashAttn.apply(q, k_raw, k_comp, sinks, window, m, block_M, block_N)
