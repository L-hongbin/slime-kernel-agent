r"""LoRA wiring for the V4 mcore model (megatron.bridge.peft.LoRA), TP=PP=EP=1.

slime has NO built-in LoRA (only regex freeze via --freeze/-only-train-params), so the
adapter wiring lives here and is invoked from the V4 model provider.

Target set (the SFT-LoRA plan / sharding contract §7): the MLA attention linears
(q_a_proj, q_b_proj, kv_proj, o_b_proj) + the compressor kv/gate projections. Experts +
routers are OFF.  **`o_a_proj` is EXCLUDED** — it is a `DeepseekV4GroupedLinear`
(block-diagonal bmm); the generic `LinearAdapter` would treat its `[8192,4096]` weight as
one dense linear and inject a rank-r adapter that ignores the 8-group structure -> wrong
math (codex flag; contract §1f/§7).

Matching (megatron.bridge.peft.ModuleMatcher.match): each target matches a module if the
leaf `name == pattern` OR `wildcard_match(pattern, full_name)` (anchored regex, `*`->`.*`).
We use **full-name wildcards** (not bare leaf names) to disambiguate:
  * `*.self_attn.kv_proj` matches ONLY the attention kv_proj, NOT `*.compressor.kv_proj`
    (the regex `^.*\.self_attn\.kv_proj$` does not match `...self_attn.compressor.kv_proj`).
  * `*.compressor.gate_proj` matches ONLY the compressor gate, NOT the shared-expert
    `shared_experts.gate_proj` (experts stay frozen).
  * `o_a_proj` is simply absent from the list -> never wrapped.

LoRA modules are plain `nn.Linear` in this model, so `LoRA.transform` wraps them with
`LinearAdapter` (lora_layers.py:139,153) — which needs no `module.config` and zero-inits
`linear_out`, so the adapted forward is the IDENTITY at init (forward parity preserved).

`LoRA.__call__` (PEFT base) FREEZES the whole model first (all `requires_grad=False`) then
swaps in adapters whose A/B params are trainable — so the base is frozen automatically.
"""

# the V4 attribute paths (full-name wildcards; see module docstring for why scoped).
V4_LORA_TARGET_MODULES = [
    "*.self_attn.q_a_proj",
    "*.self_attn.q_b_proj",
    "*.self_attn.kv_proj",
    "*.self_attn.o_b_proj",
    "*.self_attn.compressor.kv_proj",
    "*.self_attn.compressor.gate_proj",
]
# Explicitly NOT targeted (documented so the exclusion is auditable):
#   *.self_attn.o_a_proj      (grouped block-diagonal bmm — generic LoRA corrupts it)
#   *.mlp.experts.*           (routed experts frozen)
#   *.mlp.gate                (router frozen)
#   *.mlp.shared_experts.*    (shared MLP frozen)


def apply_v4_lora(model, *, dim=16, alpha=32, dropout=0.0, target_modules=None):
    """Freeze the base + add LoRA adapters to the V4 attention/compressor linears.

    Returns the transformed model (LoRA.__call__ mutates in place + returns it).  Call
    this AFTER the model is built and weights are loaded, BEFORE the optimizer is built.
    """
    from megatron.bridge.peft.lora import LoRA

    targets = target_modules if target_modules is not None else V4_LORA_TARGET_MODULES
    lora = LoRA(
        target_modules=list(targets),
        dim=dim,
        alpha=alpha,
        dropout=dropout,
    )
    model = lora(model, training=True)
    return model


def audit_lora(model):
    """Return a dict summarizing the LoRA wiring for validation/logging:
    wrapped: sorted FQNs of modules that got a LinearAdapter
    o_a_wrapped: any o_a_proj got wrapped (MUST be empty)
    n_trainable / n_frozen: param counts
    trainable_are_lora_only: every requires_grad=True param name contains 'linear_in'
      or 'linear_out' (the LoRA A/B) — i.e. no base param is trainable.
    """
    from megatron.bridge.peft.lora_layers import LinearAdapter

    wrapped = []
    for name, m in model.named_modules():
        if isinstance(m, LinearAdapter):
            wrapped.append(name)
    o_a_wrapped = [w for w in wrapped if "o_a_proj" in w]

    n_trainable = n_frozen = 0
    trainable_names = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            n_trainable += p.numel()
            trainable_names.append(n)
        else:
            n_frozen += p.numel()
    # LoRA A/B live as `.linear_in` / `.linear_out` (LinearAdapter) on the wrapped module.
    trainable_are_lora_only = all(("linear_in" in n or "linear_out" in n) for n in trainable_names)
    return dict(
        wrapped=sorted(wrapped),
        o_a_wrapped=o_a_wrapped,
        n_trainable=n_trainable,
        n_frozen=n_frozen,
        trainable_names=sorted(trainable_names),
        trainable_are_lora_only=trainable_are_lora_only,
    )


__all__ = ["apply_v4_lora", "audit_lora", "V4_LORA_TARGET_MODULES"]
