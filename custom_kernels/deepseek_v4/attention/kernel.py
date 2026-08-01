"""DeepSeek-V4-Flash core attention -- tilelang forward + hand-written backward.

Implements the A1 kernel from `handoffs/deepseek-v4/dsv4_kernel_inventory.md`:
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

import tilelang
import tilelang.language as T
import torch

NEG = -1e30  # finite stand-in for -inf masked logits (avoids inf-inf NaNs)


def _ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


# --------------------------------------------------------------------------- #
# Forward kernel                                                              #
# --------------------------------------------------------------------------- #
def _build_fwd(
    B,
    H,
    S,
    Tcomp,
    Sp,
    Tcp,
    window,
    m,
    D,
    block_M=64,
    block_N=64,
    q_pos0=0,
    raw_halo=0,
    threads=128,
    stages=1,
    dtype="bfloat16",
):
    scaling = float(D**-0.5)
    accum = "float32"
    n_q = Sp // block_M
    Sr = raw_halo + Sp  # raw axis: prepended halo rows ++ padded local raw (CP: raw_halo=0 on rank0)
    KV = Sr + Tcp  # concatenated raw (halo ++ raw_pad) ++ comp
    raw_blocks = Sr // block_N  # number of raw kv blocks (Sr % block_N == 0)

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

            # banded raw block range [raw_lo, raw_hi] (sliding window). raw axis row r
            # (r = base+j) has global pos q_pos0 - raw_halo + r; query (q0+i) has global
            # pos q_pos0 + q0 + i, so the halo shifts the raw band by +raw_halo (q_pos0
            # cancels in the raw axis).
            raw_lo = T.max(0, (q0 + raw_halo - (window - 1)) // block_N)
            raw_hi = (q0 + block_M - 1 + raw_halo) // block_N  # inclusive
            n_raw = raw_hi + 1 - raw_lo
            # compressed block count below the block's causal threshold (global qpos)
            if Tcomp > 0:
                comp_blocks = T.ceildiv(T.min((q_pos0 + q0 + block_M) // m, Tcomp), block_N)
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
                    # const bounds). raw: sliding-window causal (halo-shifted, q_pos0
                    # cancels); comp: causal threshold (w+1)*m <= qpos_global+1, comp
                    # entry w = base+j-Sr.
                    acc_s[i, j] = T.if_then_else(
                        T.if_then_else(
                            t < n_raw,
                            (base + j < raw_halo + S)
                            and (base + j <= q0 + i + raw_halo)
                            and (q0 + i - (base + j) + raw_halo < window),
                            (base + j - Sr < Tcomp) and ((base + j - Sr + 1) * m <= q_pos0 + q0 + i + 1),
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
# Sparse forward: identical to `_build_fwd` but the compressed region gains a  #
# per-(batch,query) top-k selection mask (CSA lightning indexer). A compressed #
# key is valid iff it is BOTH below the causal threshold AND selected. Raw     #
# keys are unaffected. `Mask` is uint8 [B, Sp, Tcp] (1=selected, 0=masked),    #
# padded from the real [B, S, Tcomp] top-k mask (padded rows/cols are 0).      #
#                                                                              #
# NOTE: `T.if_then_else` is a SELECT (both operands are evaluated), so the     #
# compressed sub-expr -- incl. the `Mask[...]` load -- runs even on RAW blocks #
# where `base + j - Sr < 0`. The `T.max(0, ...)` clamp keeps that read in      #
# bounds (its value is discarded by the outer select for raw blocks).         #
def _build_fwd_sparse(
    B,
    H,
    S,
    Tcomp,
    Sp,
    Tcp,
    window,
    m,
    D,
    block_M=64,
    block_N=64,
    q_pos0=0,
    raw_halo=0,
    threads=128,
    stages=1,
    dtype="bfloat16",
):
    scaling = float(D**-0.5)
    accum = "float32"
    n_q = Sp // block_M
    Sr = raw_halo + Sp
    KV = Sr + Tcp
    raw_blocks = Sr // block_N

    @T.prim_func
    def kernel(
        Q: T.Tensor([B, H, Sp, D], dtype),
        KVt: T.Tensor([B, 1, KV, D], dtype),
        Sinks: T.Tensor([H], accum),
        Mask: T.Tensor([B, Sp, Tcp], "uint8"),
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

            raw_lo = T.max(0, (q0 + raw_halo - (window - 1)) // block_N)
            raw_hi = (q0 + block_M - 1 + raw_halo) // block_N
            n_raw = raw_hi + 1 - raw_lo
            if Tcomp > 0:
                comp_blocks = T.ceildiv(T.min((q_pos0 + q0 + block_M) // m, Tcomp), block_N)
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
                    # Structural mask only (raw: sliding-window causal, halo-shifted;
                    # comp: causal threshold, global qpos). The top-k selection is
                    # applied in a SEPARATE loop below that runs only on compressed
                    # blocks (`if t >= n_raw`), so raw blocks pay no `Mask` load.
                    acc_s[i, j] = T.if_then_else(
                        T.if_then_else(
                            t < n_raw,
                            (base + j < raw_halo + S)
                            and (base + j <= q0 + i + raw_halo)
                            and (q0 + i - (base + j) + raw_halo < window),
                            (base + j - Sr < Tcomp) and ((base + j - Sr + 1) * m <= q_pos0 + q0 + i + 1),
                        ),
                        acc_s[i, j] * scaling,
                        NEG,
                    )
                if t >= n_raw:  # compressed block: AND in the top-k selection
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(Mask[bz, q0 + i, base + j - Sr] > 0, acc_s[i, j], NEG)
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
    B,
    H,
    S,
    Tcomp,
    Sp,
    Tcp,
    window,
    m,
    D,
    block_M=64,
    block_N=64,
    q_pos0=0,
    raw_halo=0,
    threads=256,
    stages=1,
    dtype="bfloat16",
):
    scaling = float(D**-0.5)
    accum = "float32"
    n_q = Sp // block_M
    Sr = raw_halo + Sp
    KV = Sr + Tcp
    raw_blocks = Sr // block_N
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

        raw_lo = T.max(0, (q0 + raw_halo - (window - 1)) // block_N)
        raw_hi = (q0 + block_M - 1 + raw_halo) // block_N
        n_raw = raw_hi + 1 - raw_lo
        if Tcomp > 0:
            comp_blocks = T.ceildiv(T.min((q_pos0 + q0 + block_M) // m, Tcomp), block_N)
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
                        (base + j < raw_halo + S)
                        and (base + j <= q0 + i + raw_halo)
                        and (q0 + i - (base + j) + raw_halo < window),
                        (base + j - Sr < Tcomp) and ((base + j - Sr + 1) * m <= q_pos0 + q0 + i + 1),
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
        raw_lo = T.max(0, (q0 + raw_halo - (window - 1)) // block_N)
        raw_hi = (q0 + block_M - 1 + raw_halo) // block_N
        n_raw = raw_hi + 1 - raw_lo
        if Tcomp > 0:
            comp_blocks = T.ceildiv(T.min((q_pos0 + q0 + block_M) // m, Tcomp), block_N)
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
# Warp-specialized split-D forward, SPARSE variant. Only the producer/consumer #
# WG0 (which computes the probs) needs the top-k mask; WG1 consumes the probs  #
# WG0 publishes to `P_sh`, so it is byte-identical to the dense `_wg1`.        #
# --------------------------------------------------------------------------- #
def _build_fwd_ws_sparse(
    B,
    H,
    S,
    Tcomp,
    Sp,
    Tcp,
    window,
    m,
    D,
    block_M=64,
    block_N=64,
    q_pos0=0,
    raw_halo=0,
    threads=256,
    stages=1,
    dtype="bfloat16",
):
    scaling = float(D**-0.5)
    accum = "float32"
    n_q = Sp // block_M
    Sr = raw_halo + Sp
    KV = Sr + Tcp
    raw_blocks = Sr // block_N
    Dh = D // 2
    BA, BB, BC, BD = 1, 2, 3, 4

    @T.macro
    def _wg0(Q, KVt, Sinks, Mask, Out, Lse, Q0_sh, Q1_sh, V0_sh, V1_sh, P_sh, sc_sh, fsc_sh, bx, by, bz):
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

        raw_lo = T.max(0, (q0 + raw_halo - (window - 1)) // block_N)
        raw_hi = (q0 + block_M - 1 + raw_halo) // block_N
        n_raw = raw_hi + 1 - raw_lo
        if Tcomp > 0:
            comp_blocks = T.ceildiv(T.min((q_pos0 + q0 + block_M) // m, Tcomp), block_N)
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
            T.gemm(Q1_sh, V1_sh, acc_s, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                # Structural mask only; top-k selection applied below on comp blocks.
                acc_s[i, j] = T.if_then_else(
                    T.if_then_else(
                        t < n_raw,
                        (base + j < raw_halo + S)
                        and (base + j <= q0 + i + raw_halo)
                        and (q0 + i - (base + j) + raw_halo < window),
                        (base + j - Sr < Tcomp) and ((base + j - Sr + 1) * m <= q_pos0 + q0 + i + 1),
                    ),
                    acc_s[i, j] * scaling,
                    NEG,
                )
            if t >= n_raw:  # compressed block: AND in the top-k selection
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(Mask[bz, q0 + i, base + j - Sr] > 0, acc_s[i, j], NEG)
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
            T.copy(acc_cast, P_sh)
            T.copy(scale_f, sc_sh)
            T.sync_threads(BB, 256)
            for i, j in T.Parallel(block_M, Dh):
                acc_o[i, j] = acc_o[i, j] * scale_f[i]
            T.gemm(acc_cast, V0_sh, acc_o)
            T.sync_threads(BC, 256)

        for i in T.Parallel(block_M):
            sink_v = Sinks[by]
            m_new = T.max(m_i[i], sink_v)
            corr = T.if_then_else(m_i[i] <= NEG, 0.0, T.exp(m_i[i] - m_new))
            l_new = l_i[i] * corr + T.exp(sink_v - m_new)
            scale_f[i] = corr / l_new
            Lse[bz, by, q0 + i] = m_new + T.log(l_new)
        T.copy(scale_f, fsc_sh)
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
        raw_lo = T.max(0, (q0 + raw_halo - (window - 1)) // block_N)
        raw_hi = (q0 + block_M - 1 + raw_halo) // block_N
        n_raw = raw_hi + 1 - raw_lo
        if Tcomp > 0:
            comp_blocks = T.ceildiv(T.min((q_pos0 + q0 + block_M) // m, Tcomp), block_N)
        else:
            comp_blocks = 0
        n_total = n_raw + comp_blocks

        for t in T.serial(0, n_total):
            T.sync_threads(BB, 256)
            T.copy(P_sh, acc_cast)
            T.copy(sc_sh, sc)
            for i, j in T.Parallel(block_M, Dh):
                acc_o[i, j] = acc_o[i, j] * sc[i]
            T.gemm(acc_cast, V1_sh, acc_o)
            T.sync_threads(BC, 256)

        T.sync_threads(BD, 256)
        T.copy(fsc_sh, fsc)
        for i, j in T.Parallel(block_M, Dh):
            acc_o[i, j] = acc_o[i, j] * fsc[i]
        T.copy(acc_o, Out[bz, by, q0 : q0 + block_M, Dh:D])

    @T.prim_func
    def kernel(
        Q: T.Tensor([B, H, Sp, D], dtype),
        KVt: T.Tensor([B, 1, KV, D], dtype),
        Sinks: T.Tensor([H], accum),
        Mask: T.Tensor([B, Sp, Tcp], "uint8"),
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
                _wg0(Q, KVt, Sinks, Mask, Out, Lse, Q0_sh, Q1_sh, V0_sh, V1_sh, P_sh, sc_sh, fsc_sh, bx, by, bz)
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
    B,
    H,
    S,
    Tcomp,
    Sp,
    Tcp,
    window,
    m,
    D,
    block_M=64,
    block_N=64,
    q_pos0=0,
    raw_halo=0,
    threads=128,
    stages=1,
    dtype="bfloat16",
):
    scaling = float(D**-0.5)
    accum = "float32"
    n_q = Sp // block_M
    Sr = raw_halo + Sp
    KV = Sr + Tcp
    raw_blocks = Sr // block_N

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

            raw_lo = T.max(0, (q0 + raw_halo - (window - 1)) // block_N)
            raw_hi = (q0 + block_M - 1 + raw_halo) // block_N
            n_raw = raw_hi + 1 - raw_lo
            if Tcomp > 0:
                comp_blocks = T.ceildiv(T.min((q_pos0 + q0 + block_M) // m, Tcomp), block_N)
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
                            (base + j < raw_halo + S)
                            and (base + j <= q0 + i + raw_halo)
                            and (q0 + i - (base + j) + raw_halo < window),
                            (base + j - Sr < Tcomp) and ((base + j - Sr + 1) * m <= q_pos0 + q0 + i + 1),
                        ),
                        T.exp(scaling * s[i, j] - lse_f[i]),
                        0.0,
                    )
                    dS_cast[i, j] = scaling * (p * (dp[i, j] - delta_f[i]))
                T.gemm(dS_cast, K_sh, dq_acc)  # dQ += scaling*dS @ K
            T.copy(dq_acc, dQ[bz, by, q0 : q0 + block_M, :])

    return tilelang.compile(kernel, out_idx=[-1])


# --------------------------------------------------------------------------- #
# Backward: dQ kernel, SPARSE variant. Masked-out compressed keys get p=0, so  #
# they contribute nothing to dQ (and produce no k-grad in the dKV kernels).    #
# Same T.max(0,...) clamp rationale as `_build_fwd_sparse`.                     #
def _build_bwd_dq_sparse(
    B,
    H,
    S,
    Tcomp,
    Sp,
    Tcp,
    window,
    m,
    D,
    block_M=64,
    block_N=64,
    q_pos0=0,
    raw_halo=0,
    threads=128,
    stages=1,
    dtype="bfloat16",
):
    scaling = float(D**-0.5)
    accum = "float32"
    n_q = Sp // block_M
    Sr = raw_halo + Sp
    KV = Sr + Tcp
    raw_blocks = Sr // block_N

    @T.prim_func
    def kernel(
        Q: T.Tensor([B, H, Sp, D], dtype),
        KVt: T.Tensor([B, 1, KV, D], dtype),
        dO: T.Tensor([B, H, Sp, D], dtype),
        Lse: T.Tensor([B, H, Sp], accum),
        Delta: T.Tensor([B, H, Sp], accum),
        Mask: T.Tensor([B, Sp, Tcp], "uint8"),
        dQ: T.Tensor([B, H, Sp, D], accum),
    ):
        with T.Kernel(n_q, H, B, threads=threads) as (bx, by, bz):
            Q_sh = T.alloc_shared([block_M, D], dtype)
            K_sh = T.alloc_shared([block_N, D], dtype)
            dO_sh = T.alloc_shared([block_M, D], dtype)
            s = T.alloc_fragment([block_M, block_N], accum)
            dp = T.alloc_fragment([block_M, block_N], accum)
            p_frag = T.alloc_fragment([block_M, block_N], accum)
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

            raw_lo = T.max(0, (q0 + raw_halo - (window - 1)) // block_N)
            raw_hi = (q0 + block_M - 1 + raw_halo) // block_N
            n_raw = raw_hi + 1 - raw_lo
            if Tcomp > 0:
                comp_blocks = T.ceildiv(T.min((q_pos0 + q0 + block_M) // m, Tcomp), block_N)
            else:
                comp_blocks = 0
            n_total = n_raw + comp_blocks

            for t in T.Pipelined(0, n_total, num_stages=stages):
                is_raw = t < n_raw
                blk = T.if_then_else(is_raw, raw_lo + t, raw_blocks + (t - n_raw))
                base = blk * block_N
                T.copy(KVt[bz, 0, base : base + block_N, :], K_sh)
                T.clear(s)
                T.gemm(Q_sh, K_sh, s, transpose_B=True)
                T.clear(dp)
                T.gemm(dO_sh, K_sh, dp, transpose_B=True)
                for i, j in T.Parallel(block_M, block_N):
                    # Structural mask only; top-k applied below on compressed blocks.
                    p_frag[i, j] = T.if_then_else(
                        T.if_then_else(
                            t < n_raw,
                            (base + j < raw_halo + S)
                            and (base + j <= q0 + i + raw_halo)
                            and (q0 + i - (base + j) + raw_halo < window),
                            (base + j - Sr < Tcomp) and ((base + j - Sr + 1) * m <= q_pos0 + q0 + i + 1),
                        ),
                        T.exp(scaling * s[i, j] - lse_f[i]),
                        0.0,
                    )
                if t >= n_raw:  # compressed block: zero p for masked-out keys
                    for i, j in T.Parallel(block_M, block_N):
                        p_frag[i, j] = T.if_then_else(Mask[bz, q0 + i, base + j - Sr] > 0, p_frag[i, j], 0.0)
                for i, j in T.Parallel(block_M, block_N):
                    dS_cast[i, j] = scaling * (p_frag[i, j] * (dp[i, j] - delta_f[i]))
                T.gemm(dS_cast, K_sh, dq_acc)
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
    B,
    H,
    S,
    Tcomp,
    Sp,
    Tcp,
    window,
    m,
    D,
    block_M=64,
    block_N=64,
    q_pos0=0,
    raw_halo=0,
    threads=128,
    stages=1,
    dtype="bfloat16",
):
    scaling = float(D**-0.5)
    accum = "float32"
    n_q = Sp // block_M
    Sr = raw_halo + Sp
    KV = Sr + Tcp
    raw_blocks = Sr // block_N

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

            # banded sliding-window query range for this raw kv block. raw axis row
            # (kv0+j) has global pos q_pos0 - raw_halo + (kv0+j); query (q0+i) has
            # global pos q_pos0 + q0 + i.  Causal: q0+i >= kv0+j - raw_halo -> q_lo;
            # window: q0+i <= kv0+j + (window-1) - raw_halo -> q_hi (q_pos0 cancels).
            q_lo = T.max(0, kv0 - raw_halo) // block_M
            q_hi = T.min(n_q - 1, (kv0 + block_N - 1 + window - 1 - raw_halo) // block_M)

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
                        (kv0 + j < raw_halo + S)
                        and (kv0 + j <= q0 + i + raw_halo)
                        and (q0 + i - (kv0 + j) + raw_halo < window),
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
    B,
    H,
    S,
    Tcomp,
    Sp,
    Tcp,
    window,
    m,
    D,
    block_M=64,
    block_N=64,
    q_pos0=0,
    raw_halo=0,
    threads=128,
    stages=1,
    dtype="bfloat16",
):
    scaling = float(D**-0.5)
    accum = "float32"
    n_q = Sp // block_M
    Sr = raw_halo + Sp
    KV = Sr + Tcp
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
            kv0 = Sr + wc0  # offset into the concatenated KV buffer (comp starts at Sr)
            T.copy(KVt[bz, 0, kv0 : kv0 + block_N, :], Kj_sh)
            T.fill(dkv_acc, 0.0)

            # TIGHT lower query bound: min entry in block is wc0, attended only by GLOBAL
            # query pos p_q >= (wc0+1)*m - 1, i.e. LOCAL i >= (wc0+1)*m - 1 - q_pos0.
            # All later queries attend (more entries valid as i grows), so q_hi = n_q-1.
            # Clamp to n_q so a fully-padded compressed block (wc0 >= Tcomp, only possible
            # when Tcomp % block_N != 0) yields a zero-extent loop instead of a negative one.
            q_lo = T.min(n_q, T.max(0, (wc0 + 1) * m - 1 - q_pos0) // block_M)

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
                        (wc0 + j < Tcomp) and ((wc0 + j + 1) * m <= q_pos0 + q0 + i + 1),
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
# Backward: compressed dKV, SPARSE variant. Masked-out compressed keys get p=0 #
# so d(kv) for them is exactly zero. Compressed-only kernel: `wc0 + j` is       #
# always in [0, Tcp), so no index clamp is needed (unlike fwd/dq, which walk    #
# raw blocks through the same select).                                          #
def _build_bwd_dkv_comp_sparse(
    B,
    H,
    S,
    Tcomp,
    Sp,
    Tcp,
    window,
    m,
    D,
    block_M=64,
    block_N=64,
    q_pos0=0,
    raw_halo=0,
    threads=128,
    stages=1,
    dtype="bfloat16",
):
    scaling = float(D**-0.5)
    accum = "float32"
    n_q = Sp // block_M
    Sr = raw_halo + Sp
    KV = Sr + Tcp
    comp_blocks = Tcp // block_N

    @T.prim_func
    def kernel(
        Q: T.Tensor([B, H, Sp, D], dtype),
        KVt: T.Tensor([B, 1, KV, D], dtype),
        dO: T.Tensor([B, H, Sp, D], dtype),
        Lse: T.Tensor([B, H, Sp], accum),
        Delta: T.Tensor([B, H, Sp], accum),
        Mask: T.Tensor([B, Sp, Tcp], "uint8"),
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

            wc0 = bx * block_N
            kv0 = Sr + wc0  # comp region starts at Sr under the halo'd raw axis
            T.copy(KVt[bz, 0, kv0 : kv0 + block_N, :], Kj_sh)
            T.fill(dkv_acc, 0.0)

            q_lo = T.min(n_q, T.max(0, (wc0 + 1) * m - 1 - q_pos0) // block_M)

            for tq in T.Pipelined(q_lo, n_q, num_stages=stages):
                q0 = tq * block_M
                T.copy(Q[bz, by, q0 : q0 + block_M, :], Q_sh)
                T.copy(dO[bz, by, q0 : q0 + block_M, :], dO_sh)
                T.copy(Lse[bz, by, q0 : q0 + block_M], lse_f)
                T.copy(Delta[bz, by, q0 : q0 + block_M], delta_f)
                T.clear(s)
                T.gemm(Kj_sh, Q_sh, s, transpose_B=True)
                T.clear(dp)
                T.gemm(Kj_sh, dO_sh, dp, transpose_B=True)
                for j, i in T.Parallel(block_N, block_M):
                    p = T.if_then_else(
                        (wc0 + j < Tcomp)
                        and ((wc0 + j + 1) * m <= q_pos0 + q0 + i + 1)
                        and (Mask[bz, q0 + i, wc0 + j] > 0),
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
# The validated warp-specialized split-D forward is the sole production path.
_FWD_CACHE: dict = {}
_DQ_CACHE: dict = {}
_DKV_RAW_CACHE: dict = {}
_DKV_COMP_CACHE: dict = {}
# Sparse (top-k over compressed) variants. dKV-raw is shared with the dense path
# (raw keys are never top-k masked), so there is no sparse raw cache.
_FWD_SPARSE_CACHE: dict = {}
_DQ_SPARSE_CACHE: dict = {}
_DKV_COMP_SPARSE_CACHE: dict = {}


def _get(cache, builder, key, *args):
    k = cache.get(key)
    if k is None:
        k = builder(*args)
        cache[key] = k
    return k


class _V4FlashAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k_raw, k_comp, sinks, window, m, comp_topk_mask, block_M, block_N, q_pos0=0, raw_halo=0):
        # q [B,H,S,D] (local queries); k_raw [B,1,raw_halo+S,D] (prepended CP halo ++
        # local raw); k_comp [B,1,Tcomp,D] or None (GLOBAL comp axis); sinks [H].
        # comp_topk_mask [B,S,Tcomp] bool or None: CSA lightning-indexer top-k
        # selection over compressed KV (None -> dense-over-compressed).
        # CP2: q_pos0 = global position of local query 0; raw_halo = halo raw rows
        # prepended to the raw axis (0 on rank0 / non-CP; 128 for CP rank>0). Defaults
        # (0, 0) reproduce the whole-sequence kernel exactly.
        B, H, S, D = q.shape
        Tcomp = 0 if k_comp is None else k_comp.shape[2]
        Sp = _ceil(S, max(block_M, block_N)) * max(block_M, block_N)
        Sr = raw_halo + Sp  # halo'd raw axis length (comp starts at axis offset Sr)
        assert raw_halo % block_N == 0, f"raw_halo {raw_halo} must be a multiple of block_N {block_N}"
        assert Sr % block_N == 0, f"raw_halo+Sp ({Sr}) must be a multiple of block_N {block_N}"
        assert k_raw.shape[2] == raw_halo + S, f"k_raw seq len {k_raw.shape[2]} must equal raw_halo+S ({raw_halo}+{S})"
        Tcp = _ceil(Tcomp, block_N) * block_N if Tcomp > 0 else 0
        KV = Sr + Tcp
        dev, dt = q.device, q.dtype
        sparse = comp_topk_mask is not None and Tcomp > 0

        q_pad = q.new_zeros(B, H, Sp, D)
        q_pad[:, :, :S] = q
        kvt = q.new_zeros(B, 1, KV, D)
        kvt[:, :, : raw_halo + S] = k_raw  # halo rows ++ local raw (padding rows stay 0)
        if Tcomp > 0:
            kvt[:, :, Sr : Sr + Tcomp] = k_comp
        sinks_f = sinks.float().contiguous()

        # Top-k mask padded to the kernel's [B, Sp, Tcp] KV axis; padded rows/cols
        # are 0 (excluded), matching the structural mask which already drops them.
        mask_pad = None
        if sparse:
            mask_pad = torch.zeros(B, Sp, Tcp, dtype=torch.uint8, device=dev)
            mask_pad[:, :S, :Tcomp] = comp_topk_mask.to(torch.uint8)

        # Warp-specialized split-D forward (2 consumer warpgroups; ~1.3x dense,
        # ~1.1x sparse vs the single-warpgroup _build_fwd). Produces byte-identical
        # Lse/Out so the hand-written backward is unchanged.
        # Cache key MUST include (q_pos0, raw_halo): building >1 CP variant in one
        # process otherwise silently reuses the first variant's kernel (design M2).
        key = (B, H, S, Tcomp, Sp, Tcp, window, m, D, block_M, block_N, str(dt), q_pos0, raw_halo)
        if sparse:
            fwd = _get(
                _FWD_SPARSE_CACHE,
                _build_fwd_ws_sparse,
                key,
                B,
                H,
                S,
                Tcomp,
                Sp,
                Tcp,
                window,
                m,
                D,
                block_M,
                block_N,
                q_pos0,
                raw_halo,
            )
            out_pad, lse_pad = fwd(q_pad, kvt, sinks_f, mask_pad)
        else:
            fwd = _get(
                _FWD_CACHE,
                _build_fwd_ws,
                key,
                B,
                H,
                S,
                Tcomp,
                Sp,
                Tcp,
                window,
                m,
                D,
                block_M,
                block_N,
                q_pos0,
                raw_halo,
            )
            out_pad, lse_pad = fwd(q_pad, kvt, sinks_f)

        ctx.save_for_backward(q_pad, kvt, sinks_f, out_pad, lse_pad)
        ctx.mask_pad = mask_pad  # non-differentiable; stashed for the sparse bwd
        ctx.sparse = sparse
        ctx.dims = (B, H, S, Tcomp, Sp, Tcp, window, m, D, block_M, block_N, q_pos0, raw_halo)
        ctx.has_comp = Tcomp > 0
        ctx.q_dtype, ctx.kr_dtype = q.dtype, k_raw.dtype
        ctx.kc_dtype = None if k_comp is None else k_comp.dtype
        return out_pad[:, :, :S]

    @staticmethod
    def backward(ctx, dout):
        q_pad, kvt, sinks_f, out_pad, lse_pad = ctx.saved_tensors
        (B, H, S, Tcomp, Sp, Tcp, window, m, D, block_M, block_N, q_pos0, raw_halo) = ctx.dims
        Sr = raw_halo + Sp
        KV = Sr + Tcp
        sparse = ctx.sparse
        mask_pad = ctx.mask_pad

        dO = q_pad.new_zeros(B, H, Sp, D)
        dO[:, :, :S] = dout
        # delta_i = sum_d dO_i . O_i   (fp32)
        delta = (dO.float() * out_pad.float()).sum(-1).contiguous()  # [B,H,Sp]

        kdq = (B, H, S, Tcomp, Sp, Tcp, window, m, D, block_M, block_N, q_pos0, raw_halo)
        # dKV-raw is unaffected by the top-k mask (raw keys are never masked), so the
        # dense raw kernel is reused on both paths.
        dkv_raw_fn = _get(
            _DKV_RAW_CACHE,
            _build_bwd_dkv_raw,
            kdq,
            B,
            H,
            S,
            Tcomp,
            Sp,
            Tcp,
            window,
            m,
            D,
            block_M,
            block_N,
            q_pos0,
            raw_halo,
        )
        dKV = torch.zeros(B, 1, KV, D, device=q_pad.device, dtype=torch.float32)

        if sparse:
            dq_fn = _get(
                _DQ_SPARSE_CACHE,
                _build_bwd_dq_sparse,
                kdq,
                B,
                H,
                S,
                Tcomp,
                Sp,
                Tcp,
                window,
                m,
                D,
                block_M,
                block_N,
                q_pos0,
                raw_halo,
            )
            dQ = dq_fn(q_pad, kvt, dO, lse_pad, delta, mask_pad)  # [B,H,Sp,D] fp32
            dkv_raw_fn(q_pad, kvt, dO, lse_pad, delta, dKV)  # raw (dense), atomic-accumulated
            if Tcp > 0:
                dkv_comp_fn = _get(
                    _DKV_COMP_SPARSE_CACHE,
                    _build_bwd_dkv_comp_sparse,
                    kdq,
                    B,
                    H,
                    S,
                    Tcomp,
                    Sp,
                    Tcp,
                    window,
                    m,
                    D,
                    block_M,
                    block_N,
                    q_pos0,
                    raw_halo,
                )
                dkv_comp_fn(q_pad, kvt, dO, lse_pad, delta, mask_pad, dKV)  # compressed, atomic-accumulated
        else:
            dq_fn = _get(
                _DQ_CACHE,
                _build_bwd_dq,
                kdq,
                B,
                H,
                S,
                Tcomp,
                Sp,
                Tcp,
                window,
                m,
                D,
                block_M,
                block_N,
                q_pos0,
                raw_halo,
            )
            dQ = dq_fn(q_pad, kvt, dO, lse_pad, delta)  # [B,H,Sp,D] fp32
            dkv_raw_fn(q_pad, kvt, dO, lse_pad, delta, dKV)  # raw, atomic-accumulated
            if Tcp > 0:
                dkv_comp_fn = _get(
                    _DKV_COMP_CACHE,
                    _build_bwd_dkv_comp,
                    kdq,
                    B,
                    H,
                    S,
                    Tcomp,
                    Sp,
                    Tcp,
                    window,
                    m,
                    D,
                    block_M,
                    block_N,
                    q_pos0,
                    raw_halo,
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
        # dk_raw covers the halo'd raw axis [0, raw_halo+S): first raw_halo rows are the
        # halo grad (the CP caller reduces them into the left neighbor's raw span), then
        # the S local rows. raw_halo=0 -> just [0, S) as before.
        dk_raw = dKV[:, :, : raw_halo + S].to(ctx.kr_dtype)
        dk_comp = None
        if ctx.has_comp:
            dk_comp = dKV[:, :, Sr : Sr + Tcomp].to(ctx.kc_dtype)
        # grads align with forward args: q, k_raw, k_comp, sinks, window, m,
        # comp_topk_mask, block_M, block_N, q_pos0, raw_halo (the mask and the two CP
        # position constants are non-differentiable).
        return dq, dk_raw, dk_comp, dsink, None, None, None, None, None, None, None


def v4flash_attention(
    q, k_raw, k_comp, sinks, window, m, block_M=64, block_N=64, comp_topk_mask=None, q_pos0=0, raw_halo=0
):
    """DeepSeek-V4-Flash core attention (training; no KV cache).

    Args:
        q:      [B, H, S, D]   post-RoPE queries (LOCAL queries under CP).
        k_raw:  [B, 1, raw_halo+S, D]   post-RoPE raw KV (K == V); under CP this is the
            prepended left halo (raw_halo rows) ++ the local raw. raw_halo=0 -> [B,1,S,D].
        k_comp: [B, 1, Tcomp, D] or None   post-RoPE compressed KV (K == V); the GLOBAL
            compressed axis (all ranks' compressed entries, allgathered) under CP.
        sinks:  [H]            per-head sink logit (frozen base param).
        window: sliding window W.
        m:      compress rate (CSA 4 / HCA 128); ignored if k_comp is None.
        comp_topk_mask: [B, S, Tcomp] bool or None. When given, the CSA lightning-
            indexer top-k selection over compressed KV: a compressed key is valid
            only if it is BOTH below the causal threshold AND selected (sparse
            attention over compressed, matching the rollout). None -> dense.
        q_pos0: global position of local query 0 (context-parallel offset; 0 for the
            whole-sequence / rank0 case). Compile-time constant threaded into the
            compressed causal threshold.
        raw_halo: number of left-halo raw rows prepended to ``k_raw`` (0 for rank0 /
            non-CP; 128 for CP rank>0 -- 127 covers the window, 128 keeps block
            alignment). Compile-time constant threaded into the raw sliding-window mask
            and the raw band ranges. Must be a multiple of block_N, as must raw_halo+Sp.

    Returns:
        out: [B, H, S, D]   attention output (conjugate-RoPE / o_proj are external).
            With q_pos0=0, raw_halo=0 this is bit-identical to the pre-CP kernel.
    """
    return _V4FlashAttn.apply(q, k_raw, k_comp, sinks, window, m, comp_topk_mask, block_M, block_N, q_pos0, raw_halo)
