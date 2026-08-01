# DS-V4 → mcore: module structure + TP/PP/EP sharding contract (DESIGN ONLY)

> The distributed-execution contract codex flagged as the prerequisite to swapping
> the M0 torch `V4Model` (`custom_kernels/deepseek_v4/megatron/`, forward-parity-validated vs HF
> at M1) for real Megatron parallel modules. **No model code here** — this fixes the
> per-weight parallel type, sharding axis, kernel placement, EP/PP plan, the
> degenerate single-GPU case, the GPTModel-compat surface, and the LoRA fit.
>
> Sources: HF `modeling_deepseek_v4.py` (5.8.1) + `configuration_deepseek_v4.py`
> (`base_model_ep_plan` ~:114); M0 modules (`attention.py`/`compressor.py`/`decoder.py`);
> Megatron-LM `megatron/core/{transformer/moe,tensor_parallel,models/gpt}`;
> `megatron.bridge.peft.lora`; slime `backends/megatron_utils/{model.py,model_provider.py}`.
> Companion plan: [[v4_megatron_phase1_plan]]; kernel map: [[dsv4_kernel_inventory]].

## 0. The supporting fact (decides everything below)

V4 is **EP-only, NO pure TP** (`base_model_ep_plan`, configuration_deepseek_v4.py:114-129;
there is *deliberately* no `base_model_tp_plan`). Reason, verbatim from the config: V4
attention is **shared-KV MQA (`num_key_value_heads=1`) + a CSA/HCA compressor branch**;
both broadcast a single KV head across all 64 query heads via `repeat_kv`, so
colwise-sharding `q_b_proj` would leave the single KV head replicated and `repeat_kv`
would no longer match the rank-local query-head count. Therefore:

- **Attention is fully replicated** across all ranks (every rank holds the entire
  attention sub-layer and computes the full head set). TP within attention = 1.
- **Routed experts are EP-sharded** on the expert axis (256 experts split across EP
  ranks), run as a grouped-GEMM, output combined by the MoE token dispatcher.
- **Shared MLP is replicated** (small, not worth TP-ing).
- **Router is replicated** (`ep_router`: each rank routes its own tokens locally, then
  dispatches).
- **mHC (A1/B2/B1) compute is replicated** because it all sits inside the replicated
  attention/residual path (see §3).

This maps cleanly onto Megatron's MoE: `expert_model_parallel_size = EP`,
`tensor_model_parallel_size = 1`, `expert_tensor_parallel_size = 1`. The only
cross-rank collectives in the steady state are the MoE token-dispatch all-to-alls
(+ DP grad all-reduce, + PP point-to-point). No TP all-reduce anywhere because TP=1.

---

## 1. Per-weight / per-module parallel contract

Conventions: `H`=hidden=4096, `Dh`=head_dim=512, `Nh`=heads=64, `qr`=q_lora_rank=1024,
`og`=o_groups=8, `or`=o_lora_rank=1024, `hc`=hc_mult=4, `E`=n_routed_experts=256,
`I`=moe_intermediate_size=2048, `V`=vocab=129280. "Replicated" = identical full weight
on every rank, no sharding, no collective. "ColumnParallel/RowParallel" only ever
shard on the **MoE TP axis (expt_tp)** for experts — for attention TP=1 so they too are
effectively replicated; we still note the *intended* class so the design is correct at
TP>1 should it ever be enabled.

### 1a. Embedding / head

| HF param | shape | mcore module | axis | notes |
|---|---|---|---|---|
| `model.embed_tokens.weight` | `[V,H]` | `LanguageModelEmbedding` (`VocabParallelEmbedding`) | vocab-parallel on TP | TP=1 → replicated, no collective. Lives on **PP first stage** only. |
| `lm_head.weight` | `[V,H]` | `output_layer` (`ColumnParallelLinear`, `gather_output`-style) | vocab-parallel on TP | TP=1 → replicated. **PP last stage** only. `tie_word_embeddings=False` (config:178) → not tied; do NOT set `share_embeddings_and_output_weights`. |

### 1b. Attention (V4Attention) — ALL replicated (TP=1, MQA constraint)

| HF param | shape | intended mcore class | axis at TP>1 | at TP=1 (this port) |
|---|---|---|---|---|
| `self_attn.q_a_proj.weight` | `[qr,H]` | ColumnParallelLinear (down-proj rank) | replicated (can't col-shard, feeds q_a_norm over full qr) | replicated nn.Linear / ColumnParallelLinear |
| `self_attn.q_a_norm.weight` | `[qr]` | replicated RMSNorm buffer | replicated | replicated (`V4RMSNorm`) |
| `self_attn.q_b_proj.weight` | `[Nh*Dh, qr]`=`[32768,1024]` | **must stay replicated** (col-shard breaks MQA `repeat_kv`) | replicated | replicated nn.Linear |
| `self_attn.q_b_norm` (unweighted) | — | replicated (no param) | replicated | replicated (`V4UnweightedRMSNorm`) |
| `self_attn.kv_proj.weight` | `[Dh,H]`=`[512,4096]` | **replicated** (single MQA head) | replicated | replicated nn.Linear |
| `self_attn.kv_norm.weight` | `[Dh]` | replicated RMSNorm | replicated | replicated |
| `self_attn.o_a_proj.weight` (grouped) | `[og*or, Nh*Dh/og]`=`[8192,4096]` viewed `[8,1024,4096]` block-diagonal bmm | **custom grouped block-diagonal linear** (NOT ColumnParallelLinear — bmm, see §1f) | replicated; if ever sharded, on the `og` group axis | replicated custom `DeepseekV4GroupedLinear` |
| `self_attn.o_b_proj.weight` | `[H, og*or]`=`[4096,8192]` | RowParallelLinear | replicated | replicated nn.Linear |
| `self_attn.sinks` | `[Nh]`=`[64]` | replicated buffer/param | replicated (per-head, full set) | replicated Parameter (frozen under LoRA) |

All attention linears are replicated in this port. The "intended mcore class" column
documents the closest stock analogue for the (unsupported) TP>1 future; **the V4 EP
plan explicitly forbids TP-ing attention**, so we never instantiate the sharded form.

### 1c. Compressors (CSA / HCA / Indexer) — replicated (sit inside replicated attention)

| HF param | shape | mcore | axis |
|---|---|---|---|
| `compressor.kv_proj.weight` | HCA `[Dh,H]`; CSA `[2*Dh,H]` | replicated nn.Linear | replicated |
| `compressor.gate_proj.weight` | HCA `[Dh,H]`; CSA `[2*Dh,H]` | replicated nn.Linear | replicated |
| `compressor.position_bias` | HCA `[m,Dh]` (m=128); CSA `[m,2*Dh]` (m=4) | replicated Parameter | replicated (frozen under LoRA) |
| `compressor.kv_norm.weight` | `[Dh]` | replicated RMSNorm | replicated |
| `compressor.rotary_emb` | buffers | replicated, non-persistent | replicated |
| `compressor.indexer.*` (CSA only) | `kv_proj[2*idh,H]`, `gate_proj[2*idh,H]`, `q_b_proj[idh*inh, qr]`, `weights_proj[inh,H]`, `position_bias[m,2*idh]`, `kv_norm[idh]` (idh=128, inh=64) | replicated (top-k **dropped** in M0; kept only for state-dict completeness, M5) | replicated |

The indexer is instantiated for HF→mcore state-dict mapping completeness but its top-k
is not applied (dense-over-compressed; tiny seq makes it a no-op — see M1_RESULTS caveats).

### 1d. mHC (V4HyperConnection ×2/layer + HyperHead) — replicated, fp32

| HF param | shape | mcore | axis |
|---|---|---|---|
| `attn_hc.fn` / `ffn_hc.fn` | `[(2+hc)*hc, hc*H]`=`[24, 16384]` | replicated Parameter (**fp32**) | replicated |
| `attn_hc.base` / `ffn_hc.base` | `[(2+hc)*hc]`=`[24]` | replicated Parameter (fp32) | replicated |
| `attn_hc.scale` / `ffn_hc.scale` | `[3]` | replicated Parameter (fp32) | replicated |
| `hc_head.hc_fn` | `[hc, hc*H]`=`[4,16384]` | replicated Parameter (fp32) | replicated |
| `hc_head.hc_base` | `[hc]`=`[4]` | replicated Parameter (fp32) | replicated |
| `hc_head.hc_scale` | `[1]` | replicated Parameter (fp32) | replicated |

mHC operates on the full `hc_mult·hidden` flattened residual stack with no head/expert
axis → nothing to shard, and B1 hardcodes `H=4/HIDDEN=4096/SINKHORN=20`. All ranks hold
identical fp32 mHC params and run the kernel locally. Must be `_keep_in_fp32_modules`.

### 1e. Router + Experts + Shared MLP (the only EP-sharded part)

| HF param | shape | mcore module | axis |
|---|---|---|---|
| `mlp.gate.weight` (TopK or Hash) | `[E,H]`=`[256,4096]` | router weight (`TopKRouter.weight` for `moe`; **custom HashRouter** for `hash_moe`) | **replicated** (`ep_router`: each rank routes locally) |
| `mlp.gate.e_score_correction_bias` (TopK only) | `[E]` | `TopKRouter.expert_bias` buffer (fp32) | replicated |
| `mlp.gate.tid2eid` (Hash only) | `[V, top_k]` long | replicated buffer (frozen lookup) | replicated |
| `mlp.experts.gate_up_proj` | `[E, 2*I, H]`=`[256,4096,4096]` | grouped-GEMM `linear_fc1` (`TEColumnParallelGroupedLinear` / `GroupedMLP.weight1`) | **EP-sharded on expert axis** → `E/EP` local experts; within each, col-parallel on expt_tp (=1 here) |
| `mlp.experts.down_proj` | `[E, H, I]`=`[256,4096,2048]` | grouped-GEMM `linear_fc2` (`TERowParallelGroupedLinear` / `GroupedMLP.weight2`) | **EP-sharded on expert axis**; row-parallel on expt_tp (=1) |
| `mlp.shared_experts.{gate,up,down}_proj.weight` | gate/up `[I,H]`, down `[H,I]` | `SharedExpertMLP` (`MLP`, `tp_group=pg.tp`) | **replicated** |

EP mechanics (Megatron `moe_layer.py`): `num_local_experts = num_moe_experts // ep_size`
(`:102`), with assertion `256 % EP == 0`; `local_expert_indices` is a contiguous block per
rank (`:108`). The grouped weights live as flat params reshaped to per-expert 3D at forward
(`GroupedMLP`: `weight1.view(num_local_experts, H, -1)`), or per-expert TE grouped linears
(`TEGroupedMLP`). HF stores `[E, 2I, H]` / `[E, H, I]`; the M3/M5 mapping slices the expert
axis to the rank's `local_expert_indices` and (for GLU) splits/reorders gate vs up halves
(`apply_swiglu_sharded_factory`). EP>1 sets `weight.allreduce=False` so expert grads are
NOT DP-all-reduced (`experts.py:90,220`).

### 1f. The grouped `o_a` is special (codex's flagged hard case)

`DeepseekV4GroupedLinear` is an `nn.Linear` subclass whose weight `[8192,4096]` is
**reinterpreted block-diagonal** `[og=8, or=1024, in=4096]` and applied as a batched
`bmm` over 8 head-groups — it is NOT a single dense GEMM and NOT a Megatron parallel
linear. Consequences:
- It stays **replicated** (TP=1) and is implemented as the custom `DeepseekV4GroupedLinear`
  (reused from HF), not `ColumnParallelLinear`.
- For LoRA it **cannot** be auto-wrapped (a `ParallelLinearAdapter` would treat the
  `[8192,4096]` weight as one dense linear and inject a rank-r adapter that ignores the
  block-diagonal structure → wrong math). It needs a **custom grouped adapter** or must be
  **excluded** from LoRA targets. See §6.
- If TP>1 were ever wanted, the natural shard is the `og` group axis (each rank owns a
  subset of the 8 groups) — but the EP plan keeps it replicated.

---

## 2. Where the 3 tilelang kernels sit (confirm: replicated compute)

All three kernels operate on **full-hidden activations inside the replicated attention/
residual path**, so every rank runs them on its full local token set. There is no kernel
that touches the expert axis, so none of them are EP-sharded.

| Kernel | call site | sharding | rank residency |
|---|---|---|---|
| **A1** `v4flash_attention(q,k_raw,k_comp,sinks,window,m)` | `V4Attention.forward` | **replicated** — attention is replicated (MQA constraint); A1 sees full `[B,Nh=64,S,Dh=512]` q + single KV head | every rank, identical |
| **B2** `hca_compress`/`csa_compress` | `V4{HCA,CSA}Compressor.forward` | **replicated** — compressor lives inside replicated attention | every rank, identical |
| **B1** `hyper_connection(_sglang)` | `V4HyperConnection.forward` (×2/layer) + analogous in `hc_head` | **replicated** — mHC mixes the full `[B,S,hc,H]` stack, fp32 | every rank, identical |

Confirmed: A1/B2/mHC are **replicated compute**. The kernels need no awareness of EP/PP/TP
— each rank invokes them on its local microbatch exactly as in the single-GPU M0 path. The
only distributed concern they introduce is **autograd input-gradient flow** (already
provided: A1→dq/dk_raw/dk_comp, B2→dkv/dgate, B1→d_x) so LoRA adapters on the surrounding
replicated linears receive gradients (the kernels hold no weights). Activation-recompute
(`recompute_granularity`) wraps the replicated module and re-invokes the deterministic
`autograd.Function.forward` — compatible without kernel changes (kernel inventory §recompute).

---

## 3. EP plan for the 256 experts

- `expert_model_parallel_size = EP`, `256 % EP == 0` → `num_local_experts = 256/EP` per rank
  (e.g. EP=8 → 32 local experts/rank).
- **Process group**: `pg_collection.ep` for the expert axis; `pg_collection.expt_tp` (=1
  here) for expert-internal TP; `pg_collection.expt_dp` for expert DP.
- **Routing**: `ep_router` — the router weight `[256,4096]` is replicated; each rank computes
  routing scores for *its own* tokens locally. For `moe` layers this is Megatron's
  `TopKRouter` with `score_function="sqrtsoftplus"`-equivalent (see §risks: sqrtsoftplus is
  NOT a stock Megatron scoring fn — stock supports only `softmax`/`sigmoid`; needs a custom
  router or score hook) and `enable_expert_bias=True` mapping `e_score_correction_bias →
  expert_bias`. For `hash_moe` layers the selection is a frozen `tid2eid[input_ids]` lookup —
  **not** expressible by `TopKRouter`; needs a **custom HashRouter** that still produces the
  learned per-expert weights but takes its indices from the table (and so needs `input_ids`
  threaded to the MoE — see §5).
- **Token dispatch (`MoEAlltoAllTokenDispatcher`)**: per forward —
  1. local `permute` by routing_map;
  2. EP **all-to-all** of tokens+probs (`token_dispatch`, `all_to_all(ep_group,...)`);
  3. (expt_tp all-gather — no-op at TP=1);
  4. grouped-GEMM expert compute on local experts;
  5. (expt_tp reduce-scatter — no-op at TP=1);
  6. inverse EP **all-to-all** (`token_combine`);
  7. local `unpermute` (topk-weighted sum) → output in original token order;
  8. **shared-expert output added** in `combine_postprocess` (overlap) or in `postprocess`
     (non-overlap).
  There is **no separate all-reduce of the routed output** across EP — the inverse all-to-all
  returns each token's contribution to its origin rank and `unpermute` does the weighted sum.
- **All-reduce points overall**: only DP grad all-reduce (standard) and PP point-to-point.
  Expert grads are excluded from the attention-DP all-reduce when EP>1 (`weight.allreduce=
  False`); they reduce over `expt_dp` instead. No TP all-reduce (TP=1).

## 4. PP plan (43 layers)

- `pipeline_model_parallel_size = PP`. Megatron splits the 43 decoder layers across PP ranks
  via `get_num_layers_to_build` / `get_transformer_layer_offset`
  (`offset = pp_rank * (num_layers // PP)` in the even case). 43 is prime → for even splits
  use uneven first/last (`num_layers_in_first_pipeline_stage` /
  `num_layers_in_last_pipeline_stage`, exposed by slime as
  `--decoder-first/last-pipeline-num-layers`) or `pipeline_model_parallel_layout`.
- **Embedding** lives on PP **first** stage (`pre_process=True`). **`hc_head` + final `norm`**
  live on PP **last** stage, applied only before the output layer (the M0 `V4Model` does
  `norm(hc_head(streams))` at the end — in PP this collapse must run on the last stage right
  before `output_layer`).
- **The `[B,S,hc_mult,H]` stream across PP boundaries (the subtle part)**: stock
  `TransformerBlock` passes a single `[S,B,H]` activation between stages via
  `set_input_tensor`. V4 threads a **4× larger `[B,S,hc,H]` stack** through every layer and
  only collapses it at `hc_head` on the last stage. Therefore the **inter-stage activation
  is the full hc-stream stack**, not a collapsed hidden state. The custom V4 block must:
  (a) on a non-first PP stage, receive the `[*, hc, H]` stack via `set_input_tensor` and feed
  it to its first local layer; (b) on a non-last PP stage, return the `[*, hc, H]` stack as
  the inter-stage activation (4× the normal P2P volume — a real cost to note); (c) place
  `hc_head` (the stream collapse) + final `norm` only on the last stage. The hc params
  (`fn/base/scale`) are per-layer and replicated, so they ride with whichever layers a PP
  rank owns; `hc_head` is a single module that only the last stage instantiates.

## 5. TP=PP=EP=1 degenerate case (the tiny SFT-LoRA sanity)

Confirmed the design runs trivially on a single GPU:
- `tensor_model_parallel_size=1`: all Megatron parallel linears short-circuit — the
  collective primitives in `tensor_parallel/mappings.py` early-return the input unchanged
  when `world_size==1` (`_reduce`/`_split`/`_gather` all `:28-86`); `output_size_per_partition
  == output_size` → a single full-weight matmul. Identical to the M0 replicated nn.Linear path.
- `pipeline_model_parallel_size=1`: `get_transformer_layer_offset` returns 0, one stage holds
  all 43 layers, embedding + `hc_head` + norm + output all co-resident. The custom block's
  `set_input_tensor` is a no-op (pre_process and post_process both True).
- `expert_model_parallel_size=1`: `num_local_experts == 256` (all experts on the one rank);
  the dispatcher all-to-all is a literal no-op (`_AllToAll.forward` returns input when
  `group.size()==1`, `mappings.py:430`); dispatch degenerates to a local permute/expert-
  compute/unpermute — i.e. exactly the HF `DeepseekV4SparseMoeBlock` grouped loop semantics.
  `weight.allreduce=True` (normal DP grad reduce).

So at 1/1/1 the whole model is the M0 torch path with Megatron wrappers that add no
collectives — this is what the tiny SFT-LoRA sanity exercises.

**What changes at EP>1** (and only then): (a) router still local, but token dispatch now
does real EP all-to-alls; (b) experts split to `256/EP` per rank; (c) expert grads reduce
over `expt_dp` not DP; (d) HF→mcore expert weight mapping must slice the expert axis per
rank and split GLU halves. Attention/compressor/mHC are unaffected (replicated). At PP>1:
the `[*,hc,H]` stack crosses stages (§4). At TP>1: **not supported** by the V4 EP plan.

## 6. GPTModel-compat (custom block exposing the interface slime/Megatron expect)

The custom V4 block cannot reuse stock `TransformerBlock`/`TransformerLayer` (single
`[S,B,H]` stream + BDA residuals; no `input_ids` to the MLP; no hc-stream). So the V4 model
is a **custom top-level `LanguageModule` subclass** that re-implements the GPTModel surface
slime drives. Required surface (from the interface audit):

- **Build hook**: register a `custom_model_provider_path` (slime `model_provider.py:72-90`)
  that returns a callable `(pre_process=True, post_process=True, vp_stage=None) -> V4ForCausalLM`.
  The provider builds a `TransformerConfig` (V4 fields: hidden, layers, EP/PP sizes,
  moe_grouped_gemm, moe_token_dispatcher_type="alltoall", num_moe_experts=256,
  moe_router_topk=6, moe_shared_expert_intermediate_size=I, etc.) — mirroring the V3 bridge
  `MLAModelProvider` config block but with V4 specifics and **no MLA/TE attention spec**.
- **Subclass `LanguageModule(MegatronModule)`**; `super().__init__(config, pg_collection)`.
  Store `self.config`, `self.pre_process`, `self.post_process`,
  `self.share_embeddings_and_output_weights=False`, `self.model_type=
  ModelType.encoder_or_decoder`.
- **`forward(self, input_ids, position_ids, attention_mask, *, labels=None,
  packed_seq_params=None, loss_mask=None, ...)`**: slime always passes `position_ids=None`,
  `attention_mask=None`, `labels=None` and a `packed_seq_params` (thd / cu_seqlens). Must
  return **fp32 logits `[b,s,V]`** when `labels is None` (slime computes loss externally),
  and raw hidden states (the `[*,hc,H]` stack — see §4) when `not post_process`. Note V4
  derives causality structurally inside A1 (sliding window) — it does not consume the
  `attention_mask`; position handling comes from `position_ids`/`packed_seq_params`.
- **`set_input_tensor(input_tensor)`** (accept tensor or 1-elem list): on non-first PP stage,
  store the incoming `[*,hc,H]` stack for the first local layer.
- **`self.output_layer`** (with `.weight`/`.bias`) on the last stage (critic-swap +
  checkpoint logic read it). Inherit `shared_embedding_or_output_weight()` and
  `compute_language_model_loss()` from `LanguageModule`; implement `sharded_state_dict()`;
  implement `build_schedule_plan()` only if combined-1f1b is enabled.
- **The custom decoder block** owns: per-PP-rank layer slice (replicate the
  `get_num_layers_to_build`/offset logic, or reuse it directly), the `[*,hc,H]` thread, the
  two mHC mix sites per layer (`post·out + comb.T@stream`), `input_ids` propagation to the
  hash-MoE router, and `hc_head`+norm on the last stage.

The MoE *sub-layer* inside each V4 layer **can** reuse Megatron's `MoELayer` (router +
grouped experts + dispatcher + shared expert) so the EP machinery is stock; only the
hash-router variant and the `input_ids` plumbing are custom. Everything outside the MoE
(attention, compressor, mHC, hc-stream) is the custom replicated path.

## 7. LoRA fit (what megatron.bridge.peft can wrap vs needs custom)

LoRA selection in `megatron.bridge.peft.LoRA` is **name-based** by default (matches
attribute FQN / wildcard against `target_modules`, default
`["linear_qkv","linear_proj","linear_fc1","linear_fc2"]`); the module *class* only picks the
adapter kind. So target_modules for V4 must name the V4 attribute paths.

| V4 module | LoRA-wrappable by megatron.bridge.peft? | how |
|---|---|---|
| `q_a_proj`, `q_b_proj`, `kv_proj`, `o_b_proj` (ColumnParallel/RowParallel/nn.Linear) | **Yes** | name them in `target_modules`; `LoRALinear`+`ParallelLinearAdapter` (or `LinearAdapter` for plain nn.Linear). nn.Linear IS wrappable. |
| compressor `kv_proj`/`gate_proj` (nn.Linear) | **Yes** (if desired) | name them; `LinearAdapter`. |
| `mlp.gate` (TopKRouter) | **Yes but opt-in** | `LoRATopKRouter`; NOT in default target list — name it explicitly only if router adaptation wanted. Hash router is custom (not a `TopKRouter`) → **not auto-wrappable**, needs custom or exclude. |
| experts `gate_up_proj`/`down_proj` (grouped-GEMM) | **Yes** (grouped path) | `is_grouped_expert_linear` FQN match → `GroupedExpertLinearAdapter`/`SharedOuterGroupedExpertAdapter`. **But base experts are frozen** in this LoRA plan → typically excluded. |
| shared MLP `gate/up/down_proj` | **Yes** | name match → adapter. |
| **`o_a_proj` (DeepseekV4GroupedLinear, block-diagonal bmm)** | **NO — needs custom** | a `ParallelLinearAdapter` would treat `[8192,4096]` as one dense linear and break the 8-group block-diagonal math. Either exclude it from LoRA or write a custom grouped-block adapter (per the M0 plan: "grouped `o_a` LoRA is not automatic"). |
| `sinks`, `position_bias`, mHC `fn/base/scale`, `hc_head.*`, `tid2eid` | N/A (params/buffers, not linears) | frozen; no adapter. A1/B2 should gate `needs_input_grad` so frozen sink/pos_bias params skip grad (B1 already does). |

The handoff's LoRA target set = MLA-style attention linears (`q_a_proj`, `q_b_proj`,
`kv_proj`, `o_b_proj`) + optionally compressor projections; **experts/routers default-off**;
**`o_a_proj` custom-or-excluded**. Because matching is name-based, the custom V4 attribute
names must be added to `target_modules` (the stock `linear_qkv`/`linear_proj` names won't
match V4's `q_a_proj`/`o_b_proj`).

---

## 8. Biggest implementation risks + open questions

1. **`scoring_func="sqrtsoftplus"` is not a stock Megatron router score function.** Megatron
   `TopKRouter` supports only `softmax`/`sigmoid` (`moe_utils.py:717-734`, else ValueError).
   V4's default is `sqrtsoftplus`. → Need a custom router score hook/subclass. Do NOT blindly
   inherit the V3 bridge's `moe_router_score_function="sigmoid"`. **Open: confirm the exact
   sqrtsoftplus + e_score_correction_bias + norm_topk_prob + routed_scaling_factor=1.5
   numerics match HF `DeepseekV4TopKRouter.forward` bit-for-bit (top-k boundary flips already
   seen at bf16, M1_RESULTS §top-k).**
2. **Hash-MoE router needs `input_ids` at the MoE sub-layer.** Stock `MoELayer`/`TopKRouter`
   take only hidden_states. → Custom `HashRouter` (frozen `tid2eid` lookup for indices,
   learned weight for scores) + thread `input_ids` from the model forward through the custom
   block to the MoE. This is the single biggest deviation from stock `MoELayer`.
3. **The `[B,S,hc,H]` inter-stage activation at PP>1 is 4× the normal P2P volume** and is the
   stream stack, not a collapsed hidden state. The custom block must own the stack-passing and
   `hc_head`-on-last-stage placement. **Open: does slime/Megatron's pipeline scheduler tolerate
   a non-`[S,B,H]` inter-stage tensor shape?** (set_input_tensor stores whatever it's given,
   but buffer-shape inference for P2P may assume `[S,B,H]`.)
4. **Grouped `o_a` LoRA + (future) sharding** — block-diagonal bmm is not a parallel linear;
   custom adapter or exclude (§1f/§7).
5. **Custom top-level model bypasses stock `TransformerBlock`** → we own `set_input_tensor`,
   PP layer offset/slice, final-norm placement, `sharded_state_dict`, and (if used)
   `build_schedule_plan`. Risk of subtle drift from Megatron infra expectations (DDP wrap,
   grad-reduce of replicated params, dist-checkpoint key layout). **Open: confirm replicated
   attention/mHC params don't get double-counted or wrongly EP-flagged in the optimizer/DDP
   grad reduction (they should reduce over the full DP group, experts over expt_dp only).**
6. **HF→mcore expert weight mapping (M3/M5)**: HF `[E,2I,H]`/`[E,H,I]` → per-rank expert-axis
   slice + GLU gate/up reorder for `TEGroupedMLP`/`GroupedMLP` (`apply_swiglu_sharded_factory`,
   and GroupedMLP's `w`/`v` split). The single MQA `kv_proj` (NOT V3's `kv_a`/`kv_b`), grouped
   `o_a`/`o_b` reshape, mHC `fn/base/scale ×2 + hc_head`, sinks, and dual routers all need a
   fresh, larger mapping table than the V3 bridge `common.py` provides.
7. **`tie_word_embeddings=False`** (config:178) → do not enable
   `share_embeddings_and_output_weights`; `lm_head` is an independent weight.
8. **Open: TE grouped-GEMM (`TEGroupedMLP`) vs legacy `GroupedMLP`.** Legacy is bf16-only and
   asserts `moe_latent_size is None` (fine for V4, no MoE latent); TE path needs TE present
   and integrates `moe_shared_expert_overlap`. The frozen-expert LoRA plan likely prefers the
   simpler path. **Decide which at M3.**
9. **`packed_seq_params` (thd) into A1.** slime passes packed cu_seqlens; A1's structural
   sliding-window/causal masks must respect document boundaries within a packed sequence.
   **Open: does A1 accept cu_seqlens / per-document masking, or does packing need to be
   disabled for the V4 sanity?** (M0 used contiguous B,S — packing is untested through A1.)

---

## codex(xhigh) 设计评审折入（2026-06-30）

全文：`../../sharding_review.md`。**方向确认正确**（EP-only + 自定义 LanguageModule），并把实现收紧。

**确认**：EP-only 对；experts 复用 mcore `MoELayer`/`GroupedMLP`(EP 下 `allreduce=False`)/`MoEAlltoAllTokenDispatcher` 可行；自定义 `LanguageModule` 是 slime 的正确接缝；**DDP/optimizer 不依赖 TransformerBlock**（按 `named_parameters`+`param.allreduce` 分桶，只要 expert 参数打对标签即可）。

**风险（收紧）**：
1. **路由必须 fork/subclass**：stock mcore 只认 softmax/sigmoid，且非 sigmoid 时拒绝 expert bias；V4 是 `sqrtsoftplus`+correction bias+hash(from `input_ids`)，而 `MoELayer.route()` 没有 `input_ids` 通路。
2. **expert 计算还不是精确 V4**：HF 在 SwiGLU 前 clamp gate/up，mcore `GroupedMLP` 直接 act → **即使 TP=1 也要自定义 expert 激活才能对齐**。
3. **PP>1 是最弱项**：调度器硬编码 `(seq,mbs,hidden)`，`[B,S,hc,H]` 流会破 PP → **PP>1 推迟**；启用时只走 non-interleaved + 显式 `adjust_tensor_shapes_fn` + 传 `[S,B,hc,H]`/flatten，**不要在 PP 边界 collapse 流**。
4. `sharded_state_dict` 是真活（PP-aware 层偏移 + expert sharding 元数据 + HF 映射）。
5. **LoRA 会悄悄毁了 grouped `o_a`**：HF `GroupedLinear` 继承 `nn.Linear` 但 forward 是块对角 bmm，bridge LoRA 会按 dense 包它 → **首轮 SFT 把 `*.o_a_proj` 排除在 LoRA 外**，用显式 target_modules、别用 all-linear 广匹配。

**实现顺序（codex 建议，采纳）**：
1. **先只做 TP=PP=EP=1**（单卡，就是 SFT sanity 用的）：自定义 GPT-like `LanguageModule`，内部 seq-major，返回 slime 兼容的 fp32 `[B,S,V]`。
2. 复用 `MoEAlltoAllTokenDispatcher` + grouped expert 布局，**最小 subclass `MoELayer`** 让 `forward(..., input_ids=...)` 把 ID 传进 V4 top-k/hash 路由（不要用旁路状态）。
3. 加 V4 expert 激活包装（精确 clamp+SwiGLU）再谈 parity。
4. 首轮 SFT 把 `*.o_a_proj` 排除出 LoRA；显式 target_modules。
5. PP>1 推迟。
6. 长跑前的 sanity 套件：HF/M0 单 batch parity、router idx/weights parity、expert 输出 parity、LoRA 包裹模块审计、`param.allreduce` 审计、`sharded_state_dict()` key 审计、packed-THD 边界、mHC fp32 参数检查。
