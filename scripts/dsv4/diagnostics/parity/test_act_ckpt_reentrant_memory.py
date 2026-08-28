#!/usr/bin/env python
"""Reproduce the ctx-16k activation-checkpoint recompute-holding memory bug and test
whether custom ``torch.autograd.Function``s aggravate it.

Background (historical snapshot: ``git show
1a123a4:handoffs/deepseek-v4/16k_memory_analysis.md``): with each V4
decoder layer wrapped in ``torch.utils.checkpoint``, the forward memory stays flat
(only layer inputs stored) but the stage BACKWARD shows a monotonic per-layer climb
-> OOM. The open question is whether ``use_reentrant=False`` (pytorch#147449) is the
sole cause, or whether the custom autograd Functions inside the checkpointed region
(our TileLang attn/compressor/mHC wrappers) pin the recompute frames.

Design: a parent orchestrator spawns one clean SUBPROCESS per matrix cell so the
CUDA allocator starts fresh each time. Each child
builds a stack of N layers, runs its own ``checkpoint`` loop, and probes
``torch.cuda.memory_allocated()`` as the backward walks the layers (a grad hook on
each layer's output tensor, which fires deepest-layer-first during backward).

Matrix = {use_reentrant: False, True} x {variant: A our-kernels, B torch-fallback,
C official-TileKernels-mHC-synthetic}.

Run the full matrix:
    CUDA_VISIBLE_DEVICES=1 python scripts/dsv4/diagnostics/parity/test_act_ckpt_reentrant_memory.py
Run one cell (used internally by the parent):
    ACTCKPT_CELL=1 VARIANT=A REENTRANT=0 CUDA_VISIBLE_DEVICES=1 python ...
"""

import json
import os
import statistics
import subprocess
import sys

GB = 1024.0**3

CKPT = "/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8"
TILEKERNELS = "/nfs/FM/chenshuailin/projects/kernel_agents/TileKernels"
REPO = "/nfs/FM/chenshuailin/projects/kernel_agents/slime-v4flash-lora"

# Modest but signal-bearing shapes. hidden/head_dim/hc_mult are structural (kernels
# hard-code them) and are NOT shrunk. Experts ARE shrunk (documented override) so the
# 8-layer stack fits GPU 1 -- the recompute-holding bug lives in attn+mHC+compressor,
# not the MoE weight footprint.
N_LAYERS = int(os.environ.get("ACTCKPT_N", "8"))
B = int(os.environ.get("ACTCKPT_B", "1"))
S = int(os.environ.get("ACTCKPT_S", "4096"))
N_EXPERTS = int(os.environ.get("ACTCKPT_EXPERTS", "8"))
MOE_INTER = int(os.environ.get("ACTCKPT_MOE_INTER", "512"))


# ======================================================================================
# CHILD: one matrix cell
# ======================================================================================
def _init_dist_1rank():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", os.environ.get("ACTCKPT_PORT", "29631"))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    import torch

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
    from megatron.core import parallel_state

    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, expert_model_parallel_size=1
        )
    from megatron.core import tensor_parallel

    tensor_parallel.model_parallel_cuda_manual_seed(0)


def _build_ab(variant, lora_dim=0):
    """Variants A/B: a stack of real V4DecoderLayers.

    For B, the torch references are injected explicitly below because production
    fixes these paths to validated kernels. ``lora_dim>0`` applies production LoRA.
    """
    import torch

    sys.path.insert(0, REPO)
    _init_dist_1rank()
    torch.manual_seed(0)

    from custom_kernels.deepseek_v4.megatron.model_provider import build_v4_mcore_model
    from transformers import AutoConfig

    if variant == "B":
        from custom_kernels.deepseek_v4.attention.reference import attention_reference
        from custom_kernels.deepseek_v4.compression.reference import csa_compress_ref, hca_compress_ref
        from custom_kernels.deepseek_v4.megatron import attention, compressor, decoder
        from custom_kernels.deepseek_v4.mhc.reference import hyper_connection_forward

        def attention_reference_for_module(q, k_raw, k_comp, sinks, window, m, comp_topk_mask=None):
            return attention_reference(q, k_raw, k_comp, sinks, window, m, comp_topk_mask=comp_topk_mask).to(q.dtype)

        attention._v4flash_attention = attention_reference_for_module
        compressor._csa_compress = csa_compress_ref
        compressor._hca_compress = hca_compress_ref
        decoder._HC = hyper_connection_forward

    cfg = AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
    cfg.num_hidden_layers = N_LAYERS
    cfg.n_routed_experts = N_EXPERTS
    cfg.num_local_experts = N_EXPERTS
    cfg.moe_intermediate_size = MOE_INTER
    cfg.num_experts_per_tok = 2

    dev = torch.device("cuda")
    model = build_v4_mcore_model(cfg, params_dtype=torch.bfloat16, init_weights=True).to(dev).to(torch.bfloat16)
    # mHC / hc_head / correction-bias kept fp32 (matches slime's Float16Module path).
    if hasattr(model, "restore_fp32_modules"):
        model.restore_fp32_modules()

    if lora_dim > 0:
        # Production LoRA path: freezes the whole base and wraps the attention/compressor
        # linears with trainable (zero-init) adapters INSIDE each checkpointed layer. This
        # is the faithful RL regime -- base frozen, only tiny adapters carry grad -- so the
        # backward series reflects held recompute buffers, not param-grad accumulation.
        from custom_kernels.deepseek_v4.megatron.lora import apply_v4_lora, audit_lora

        model = apply_v4_lora(model, dim=lora_dim, alpha=2 * lora_dim, dropout=0.0)
        # LinearAdapter A/B may be created in fp32/CPU; match the frozen base (cuda/bf16).
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.to(device=dev, dtype=torch.bfloat16)
        audit = audit_lora(model)
        print(
            f"[lora] wrapped={len(audit['wrapped'])} n_trainable={audit['n_trainable']} "
            f"lora_only={audit['trainable_are_lora_only']}",
            flush=True,
        )

    model.train()

    layers = list(model.layers)
    assert len(layers) == N_LAYERS, f"got {len(layers)} layers"

    x = torch.zeros(B, S, cfg.hidden_size, device=dev, dtype=torch.bfloat16)
    position_ids = torch.arange(S, device=dev).unsqueeze(0).expand(B, -1)
    pos_emb = model._position_embeddings(x, position_ids)
    input_ids = torch.randint(0, cfg.vocab_size, (B, S), device=dev)

    def make_hidden():
        h = torch.randn(B, S, cfg.hc_mult, cfg.hidden_size, device=dev, dtype=torch.bfloat16) * 0.1
        h.requires_grad_(True)
        return h

    def call(layer, h):
        return layer(h, pos_emb, input_ids)

    return layers, make_hidden, call


def _build_c(lora_dim=0):
    """Variant C: a synthetic layer built from DeepSeek's OFFICIAL TileKernels mHC
    autograd ops -- mhc_pre -> torch Linear -> mhc_post -- matching hc_mult=4 shapes.
    Tests whether the official custom Functions pin recompute frames like ours.
    lora_dim>0 freezes the base and adds a trainable low-rank adapter on the sublayer
    linear (mimics the frozen-base + tiny-adapter RL regime)."""
    import torch
    import torch.nn as nn

    sys.path.insert(0, TILEKERNELS)
    from tile_kernels.modeling.mhc.functional import mhc_post, mhc_pre

    torch.manual_seed(0)
    dev = torch.device("cuda")
    hidden = 4096
    mhc_mult = 4
    fn_rows = mhc_mult * (mhc_mult + 2)  # 24

    class SynthMHCLayer(nn.Module):
        def __init__(self):
            super().__init__()
            # TileKernels mHC kernels require the fn / norm weights in fp32.
            self.fn = nn.Parameter(torch.randn(fn_rows, mhc_mult * hidden, device=dev, dtype=torch.float32) * 0.02)
            self.scale = nn.Parameter(torch.ones(3, device=dev, dtype=torch.float32))
            self.base = nn.Parameter(torch.zeros(fn_rows, device=dev, dtype=torch.float32))
            self.norm_weight = nn.Parameter(torch.ones(mhc_mult * hidden, device=dev, dtype=torch.float32))
            self.lin = nn.Linear(hidden, hidden, bias=False, device=dev, dtype=torch.bfloat16)
            self.lora_a = self.lora_b = None
            if lora_dim > 0:
                # frozen base + trainable low-rank adapter (zero-init B -> identity at init).
                for p in (self.fn, self.scale, self.base, self.norm_weight):
                    p.requires_grad_(False)
                self.lin.weight.requires_grad_(False)
                self.lora_a = nn.Parameter(torch.randn(hidden, lora_dim, device=dev, dtype=torch.bfloat16) * 0.02)
                self.lora_b = nn.Parameter(torch.zeros(lora_dim, hidden, device=dev, dtype=torch.bfloat16))

        def forward(self, residual):
            layer_input, ctx = mhc_pre(
                residual, self.fn, self.scale, self.base, norm_weight=self.norm_weight, mhc_mult=mhc_mult
            )
            y = self.lin(layer_input)
            if self.lora_a is not None:
                y = y + (layer_input @ self.lora_a) @ self.lora_b
            post_mix, comb_mix = ctx
            return mhc_post(y, residual, post_mix, comb_mix)

    layers = [SynthMHCLayer() for _ in range(N_LAYERS)]

    def make_hidden():
        h = torch.randn(B, S, mhc_mult, hidden, device=dev, dtype=torch.bfloat16) * 0.1
        h.requires_grad_(True)
        return h

    def call(layer, h):
        return layer(h)

    return layers, make_hidden, call


def _run_stack(layers, make_hidden, call, reentrant, use_ddp=False):
    """One measured checkpoint fwd+bwd with a per-layer backward memory probe."""
    import torch

    # The checkpoint loop lives INSIDE the module so that, when DDP-wrapped, the
    # forward routes through DDP and its gradient-ready backward hooks are actually
    # installed -- that hook<->non-reentrant-checkpoint interaction is what
    # pytorch#147449 is about. The per-layer probe hook is registered on each layer's
    # checkpoint OUTPUT; it fires deepest-layer-first as the backward walks the stack.
    stack = _CkptStack(layers, call, reentrant).cuda()
    if use_ddp:
        stack = torch.nn.parallel.DistributedDataParallel(stack, find_unused_parameters=True)

    def one_pass(measured):
        torch.cuda.synchronize()
        h = make_hidden()
        probe = []  # (layer_index, allocated_bytes) appended in backward-firing order
        _module(stack)._probe = probe if measured else None
        out = stack(h)
        alloc_after_fwd = torch.cuda.memory_allocated()
        loss = out.float().pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        return probe, alloc_after_fwd

    # warmup (compile + allocator growth), then reset peak and measure.
    one_pass(measured=False)
    prealloc_grad = os.environ.get("ACTCKPT_PREALLOC_GRAD", "0") == "1"
    for p in stack.parameters():
        if prealloc_grad and p.grad is not None:
            # KEEP the grad buffer allocated (zero its values) so the measured backward
            # accumulates in place -- first-touch grad allocation can't confound the series.
            p.grad.zero_()
        else:
            p.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    probe, alloc_after_fwd = one_pass(measured=True)
    peak = torch.cuda.max_memory_allocated()
    return probe, alloc_after_fwd, peak, baseline


def _module(m):
    """Unwrap DDP to reach the _CkptStack."""
    return m.module if hasattr(m, "module") else m


def _classify(probe):
    """probe is (layer_idx, bytes) in backward-firing order (deepest layer first).
    CLIMBING = memory rises monotonically across the backward walk (transients held);
    FLAT = stays put (transients freed per segment)."""
    series = [m for _, m in probe]
    if len(series) < 3:
        return "INSUFFICIENT", {}
    steps = [series[k] - series[k - 1] for k in range(1, len(series))]
    n_up = sum(1 for d in steps if d > 0.05 * GB)  # count >50MB rises
    climb = series[-1] - series[0]
    rng = max(series) - min(series)
    med_abs_step = statistics.median([abs(d) for d in steps]) if steps else 0.0
    # CLIMBING: >=6 layers each add held memory AND total climb dominates the range.
    verdict = "CLIMBING" if (n_up >= 6 and climb > 0.5 * rng) else "FLAT"
    return verdict, {
        "n_up": n_up,
        "climb_gb": round(climb / GB, 3),
        "range_gb": round(rng / GB, 3),
        "med_step_gb": round(med_abs_step / GB, 4),
        "first_gb": round(series[0] / GB, 3),
        "last_gb": round(series[-1] / GB, 3),
    }


import torch.nn as _nn  # noqa: E402


class _CkptStack(_nn.Module):
    """Runs the per-layer ``torch.utils.checkpoint`` loop that production uses
    (mcore_model.py V4LanguageModel.forward), with a per-layer backward memory probe."""

    def __init__(self, layers, call, reentrant):
        super().__init__()
        self.layers = _nn.ModuleList(layers)
        self._call = call
        self._reentrant = reentrant
        self._probe = None  # set to a list to enable probing on the measured pass

    def forward(self, h):
        import torch
        from torch.utils.checkpoint import checkpoint

        def mk(idx):
            def hook(grad):
                self._probe.append((idx, torch.cuda.memory_allocated()))
                return None

            return hook

        x = h
        for i, layer in enumerate(self.layers):
            x = checkpoint(self._call, layer, x, use_reentrant=self._reentrant)
            if self._probe is not None:
                x.register_hook(mk(i))
        return x


def run_cell():
    import torch

    variant = os.environ["VARIANT"]
    reentrant = os.environ["REENTRANT"] == "1"
    use_ddp = os.environ.get("USE_DDP", "0") == "1"
    freeze = os.environ.get("ACTCKPT_FREEZE", "0") == "1"
    lora_dim = int(os.environ.get("ACTCKPT_LORA_DIM", "0") or 0)
    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES already restricts to the physical gpu

    if variant in ("A", "B"):
        layers, make_hidden, call = _build_ab(variant, lora_dim=lora_dim)
    elif variant == "C":
        layers, make_hidden, call = _build_c(lora_dim=lora_dim)
    else:
        raise ValueError(variant)

    # FROZEN mode isolates the recompute-holding bug from parameter-gradient
    # accumulation: with all layer params frozen, the ONLY thing that can grow across
    # the backward walk is HELD recompute activation transients. This also matches the
    # real run, which is LoRA (base frozen, only tiny adapters train). With params
    # trainable, per-layer param-grad accumulation swamps the signal and is
    # reentrant-invariant.
    if freeze and lora_dim == 0:
        for layer in layers:
            for p in layer.parameters():
                p.requires_grad_(False)

    n_trainable = sum(p.numel() for layer in layers for p in layer.parameters() if p.requires_grad)
    probe, alloc_after_fwd, peak, baseline = _run_stack(layers, make_hidden, call, reentrant, use_ddp=use_ddp)
    verdict, stats = _classify(probe)
    result = {
        "variant": variant,
        "reentrant": reentrant,
        "use_ddp": use_ddp,
        "freeze": freeze,
        "lora_dim": lora_dim,
        "prealloc_grad": os.environ.get("ACTCKPT_PREALLOC_GRAD", "0") == "1",
        "n_trainable_params": n_trainable,
        "n_layers": N_LAYERS,
        "B": B,
        "S": S,
        "verdict": verdict,
        "baseline_gb": round(baseline / GB, 3),
        "alloc_after_fwd_gb": round(alloc_after_fwd / GB, 3),
        "peak_gb": round(peak / GB, 3),
        "series_gb": [round(m / GB, 3) for _, m in probe],
        "layer_order": [i for i, _ in probe],
        **stats,
    }
    print("RESULT_JSON:" + json.dumps(result), flush=True)


# ======================================================================================
# PARENT: orchestrate the matrix
# ======================================================================================
def _spawn(variant, reentrant, use_ddp=False, port="29631"):
    env = dict(os.environ)
    env["ACTCKPT_CELL"] = "1"
    env["VARIANT"] = variant
    env["REENTRANT"] = "1" if reentrant else "0"
    env["USE_DDP"] = "1" if use_ddp else "0"
    env["ACTCKPT_PORT"] = port
    env.setdefault("TILELANG_CACHE_DIR", "/tmp/claude-0/actckpt/tilelang_cache")
    env.setdefault("TILELANG_TMP_DIR", "/tmp/claude-0/actckpt/tilelang_tmp")

    tag = f"{variant} reentrant={reentrant} ddp={use_ddp}"
    print(f"\n===== CELL {tag} =====", flush=True)
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__)],
        env=env,
        capture_output=True,
        text=True,
        timeout=int(os.environ.get("ACTCKPT_CELL_TIMEOUT", "1200")),
    )
    out = proc.stdout + "\n" + proc.stderr
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            result = json.loads(line[len("RESULT_JSON:") :])
    if result is None:
        tail = "\n".join(out.strip().splitlines()[-25:])
        print(f"[cell {tag}] NO RESULT (rc={proc.returncode}). tail:\n{tail}", flush=True)
        return {"variant": variant, "reentrant": reentrant, "use_ddp": use_ddp, "verdict": "ERROR", "error_tail": tail}
    print(
        f"[cell {tag}] verdict={result['verdict']} peak={result.get('peak_gb')}GB "
        f"climb={result.get('climb_gb')}GB n_up={result.get('n_up')}",
        flush=True,
    )
    return result


def main():
    results = []
    port_base = 29631
    idx = 0
    for variant in ("A", "B", "C"):
        for reentrant in (False, True):
            results.append(_spawn(variant, reentrant, port=str(port_base + idx)))
            idx += 1

    # DDP follow-up: only if a use_reentrant=False variant did NOT climb DDP-free
    # (per the task: the bug is documented to need DDP in some cases). Skipped in
    # frozen mode (single-rank DDP with zero trainable params has nothing to wrap).
    freeze_mode = os.environ.get("ACTCKPT_FREEZE", "0") == "1"
    for variant in ("A", "B") if not freeze_mode else ():
        cell = next(
            (r for r in results if r["variant"] == variant and not r["reentrant"] and not r.get("use_ddp")), None
        )
        if cell and cell.get("verdict") == "FLAT":
            print(
                f"\n[ddp-followup] {variant} use_reentrant=False was FLAT DDP-free; retesting under single-rank DDP",
                flush=True,
            )
            results.append(_spawn(variant, False, use_ddp=True, port=str(port_base + idx)))
            idx += 1

    print("\n\n================= MATRIX =================", flush=True)
    hdr = f"{'variant':<8}{'reentrant':<11}{'ddp':<6}{'verdict':<12}{'peak_GB':<10}{'climb_GB':<10}{'n_up':<6}{'fwd_GB':<9}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for r in results:
        print(
            f"{r['variant']:<8}{str(r['reentrant']):<11}{str(r.get('use_ddp', False)):<6}"
            f"{r['verdict']:<12}{str(r.get('peak_gb', '-')):<10}{str(r.get('climb_gb', '-')):<10}"
            f"{str(r.get('n_up', '-')):<6}{str(r.get('alloc_after_fwd_gb', '-')):<9}",
            flush=True,
        )
    print("\n--- per-layer backward memory series (GB, deepest-layer-first) ---", flush=True)
    for r in results:
        if r.get("series_gb"):
            print(f"{r['variant']} reentrant={r['reentrant']} ddp={r.get('use_ddp')}: {r['series_gb']}", flush=True)

    print("\nFULL_RESULTS_JSON:" + json.dumps(results), flush=True)


if __name__ == "__main__":
    if os.environ.get("ACTCKPT_CELL") == "1":
        run_cell()
    else:
        main()
