# R2 Notes — V4-Flash Native FP8 Checkpoint Mapping

> Renamed from `R2_NOTES.md` (gate-R2 era); modules renamed: `r2_checkpoint`→`native_checkpoint`, `r2_slice_torch_dist`→`slice_torch_dist`, `r2_verify_torch_dist`→`verify_torch_dist`, `r2_real_weight_parity`→`real_weight_parity`. Historical run logs keep the old names. Evidence files formerly under `handoffs/in_progress/` (verify/parity `.txt`, audit/plan `.json`) were removed in the 2026-07-03 cleanup; results are re-derivable via `scripts/dsv4/convert_torch_dist.sh` (chained verify) and `real_weight_parity.py`, and the surviving run logs live in `handoffs/deepseek-v4/r2_logs/`.

Status: mapping/dequant audit PASS; `torch_dist` preflight slice save/load PASS; full
43-layer actor `PP3_EP8` conversion/load-back PASS on node64/69/70, leaving node62
for rollout; single-layer real-weight HF eager vs mcore forward parity PASS at EP=1.

## What Was Proven

- Real checkpoint path exists: `/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8`.
- Megatron training cannot load this raw HF/native safetensors path directly. It is only the conversion input and sglang rollout base path; training needs a Megatron checkpoint, normally `torch_dist`.
- `native_checkpoint.py` maps actual native V4 keys to current mcore keys, dequantizes FP8 `*.weight + *.scale` tensors, and assembles routed expert slices.
- `V4LanguageModel` can now build an explicit subset of global layer ids while saving
  local state-dict keys with global checkpoint `ShardedTensor.key` names.
- `resolve_v4_layer_ids()` reuses Megatron Core PP helpers and passes the 43-layer PP=4
  uneven split test (`10/11/11/11` layers).
- PP x EP metadata-only conversion plan estimates were generated for high-EP-first
  6-node H20 candidates; the older PP4+EP2/EP4 files are retained as 8-card estimates,
  not recommended topology.
- Real audit evidence: `handoffs/in_progress/r2_v4flash_checkpoint_audit.json`.
- Metadata-only size evidence: `handoffs/in_progress/r2_v4flash_size_estimate.json`.
- PP x EP plan evidence:
  `handoffs/in_progress/r2_v4flash_pp1_ep16_plan.json`,
  `handoffs/in_progress/r2_v4flash_pp1_ep32_plan.json`,
  `handoffs/in_progress/r2_v4flash_pp2_ep8_plan.json`,
  `handoffs/in_progress/r2_v4flash_pp2_ep16_plan.json`,
  `handoffs/in_progress/r2_v4flash_pp3_ep8_plan.json`,
  `handoffs/in_progress/r2_v4flash_pp3_ep16_plan.json`,
  `handoffs/in_progress/r2_v4flash_pp6_ep8_plan.json`,
  `handoffs/in_progress/r2_v4flash_pp4_ep2_plan.json`,
  `handoffs/in_progress/r2_v4flash_pp4_ep4_plan.json`.
- Real-shape 1-layer `torch_dist` slice evidence:
  `handoffs/in_progress/r2_slice1_torch_dist_verify.txt`.
- Logical EP shard preflight evidence:
  `handoffs/in_progress/r2_slice_ep16_rank7_verify.txt`.
- Real 2-rank EP preflight evidence:
  `handoffs/in_progress/r2_slice_real_ep2_verify.txt`.
- Real 4-rank PP+EP preflight evidence:
  `handoffs/in_progress/r2_slice_real_pp2_ep2_verify.txt`.
- Real 8-rank high-EP preflight evidence:
  `handoffs/in_progress/r2_slice_real_ep8_verify.txt`.
- Full 43-layer actor conversion evidence:
  `handoffs/in_progress/r2_pp3_ep8_verify.txt` and local node64 conversion log
  `handoffs/deepseek-v4/r2_logs/convert_pp3_ep8_node64.log`. The node69/node70
  conversion logs live on their containers at the same repo-relative path.
- Real-weight forward parity evidence:
  `handoffs/in_progress/r2_real_weight_layer0_parity.txt`,
  `handoffs/in_progress/r2_real_weight_layer2_parity.txt`, and
  `handoffs/in_progress/r2_real_weight_layer3_parity.txt`.

## Real Checkpoint Audit

Command:

```bash
python -m custom_kernels.deepseek_v4.megatron.native_checkpoint \
  --checkpoint /nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8 \
  --sample-layers 0 2 42 \
  --sample-experts 0 255 \
  --output-json handoffs/in_progress/r2_v4flash_checkpoint_audit.json \
  --estimate-size-json handoffs/in_progress/r2_v4flash_size_estimate.json
```

Results:

- indexed keys: 69,187
- actual safetensors keys: 69,143
- shards: 46
- stale index entries: 44, all `*.wo_a.scale` (43 model layers + 1 MTP); actual `wo_a.weight` tensors are bf16, so these stale scales are ignored.
- ignored MTP keys: 1,574, because current training model does not instantiate V4 MTP.
- direct non-expert mcore-mapped keys: 1,199
- routed expert native weight keys: 33,024
- unmapped non-MTP actual keys: 0

Sample audited tensors cover embedding/head/norm/mHC, sliding layer 0, compressed layer 2, HCA/indexer layer 42, hash router `tid2eid`, learned router `e_score_correction_bias`, FP8 attention/shared-expert tensors, and expert 0/255 assembly.

## Size Estimate

The metadata-only estimator reads safetensors headers, not tensor payloads. It estimates the
mapped Megatron model payload after converting floating tensors to bf16:

- native actual payload: 273.84 GiB
- native non-MTP payload: 267.65 GiB
- ignored MTP payload: 6.19 GiB
- ignored FP8 scale payload: 0.06 GiB
- converted mcore payload: 529.63 GiB
- filesystem free at estimate time: 612.96 GiB

This excludes distributed-checkpoint container overhead, optimizer state, temporary files, and
extra replicated-rank copies. The estimate means a single-process full bf16 `torch_dist`
conversion is too close to the available `/nfs/FM` free space to launch unattended.

## PP x EP Plan Estimate

The plan estimator assigns contiguous layers to PP ranks and routed experts to contiguous
EP ranks from safetensors metadata only. The topology decision was corrected on 2026-07-01:
DeepSeek-V3's official technical report uses large EP (64-way Expert Parallelism across 8
nodes) plus PP/DualPipe, so the local plan should be high-EP-first rather than "large PP,
small EP". DeepSeek-V4's technical report keeps the same direction by emphasizing
fine-grained EP communication/scheduling. Sources:
`https://arxiv.org/html/2412.19437v1` and `https://arxiv.org/html/2606.19348v1`.

With the current cap of 6 H20 nodes (48 GPUs), generated metadata-only candidates are:

- PP1 + EP16: 16 GPUs, max per-rank materialized payload estimate 45.88 GiB. This is
  the high-EP 2-node actor target, but it OOMed on H20 during real model load.
- PP1 + EP32: 32 GPUs, max per-rank materialized payload estimate 29.75 GiB.
- PP2 + EP8: 16 GPUs, max per-rank materialized payload estimate 39.96 GiB. Lower EP
  than PP1_EP16, so it is not preferred despite lower payload.
- PP3 + EP8: 24 GPUs, max per-rank materialized payload estimate 27.54 GiB. This uses
  three actor nodes and leaves one rollout node. It is the current working 4-node
  fallback after PP1_EP16 OOM.
- PP2 + EP16: 32 GPUs, max per-rank materialized payload estimate 23.46 GiB. This uses
  all four currently available H20 nodes and therefore is conversion-only, not the
  current training resource split.
- PP3 + EP16: 48 GPUs, max per-rank materialized payload estimate 16.29 GiB. This is the
  preferred all-6-node R2 conversion candidate once the two extra H20 nodes are available.
- PP6 + EP8: 48 GPUs, max per-rank materialized payload estimate 15.13 GiB. This is a
  payload-lower comparison point, not the preferred training direction, because it moves
  toward larger PP and smaller EP.
- Historical 8-card estimates: PP4 + EP2 = 69.98 GiB/rank; PP4 + EP4 = 36.98 GiB/rank.
- Total unique converted payload remains 529.63 GiB in all plans.

This is not a full conversion proof. It excludes checkpoint container overhead, live model
allocation overhead, NCCL/DCP workspace, optimizer state, activations, rollout/KV memory, and
runtime dispatcher buffers. It does show PP-only is not the right direction; use high EP
first, adding PP only as needed to fit dense/non-expert replicated payload and pipeline
scheduling. After the completed `PP3_EP8` materialization, the next training gate is R3
EP>1 forward dispatch, not another metadata-only R2 plan.

## Full 43-Layer PP3_EP8 Actor Conversion

Current verified actor checkpoint:
`/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8-r2-pp3-ep8-torch_dist`.

- Topology: PP3 x EP8 = 24 ranks on node64/node69/node70, leaving node62 for rollout.
- Rank ownership: node64 ranks 0-7, node69 ranks 8-15, node70 ranks 16-23.
- Output size at verify time: node64 186G, node69 172G, node70 173G.
- Required metadata copy from rank0 node64 to node69/node70:
  `latest_checkpointed_iteration.txt`, `release/common.pt`, `release/metadata.json`,
  and hidden `release/.metadata`.
- Load-back verify: 24/24 ranks, `load_state_dict_missing=[]`,
  `load_state_dict_unexpected=[]`, representative `q_a_proj` and first/last local
  expert gate/down diffs all `0.0`.
- Launcher: `scripts/run.r2.v4.convert.pp3_ep8.sh`; it uses static torchrun
  rendezvous with `MASTER_ADDR`/`NODE_RANK` and `bond0` for Gloo/NCCL. The earlier
  c10d rendezvous path hung on hostname/interface selection; `PP1_EP16` then OOMed.

## Real-Weight EP1 Forward Parity

`real_weight_parity.py` builds a sliced HF eager reference and a sliced mcore
`V4LanguageModel`, loads real native FP8 weights into both, and compares per-seam forward
outputs. The first run exposed a harness bug: HF was built through `meta -> to_empty`, so
non-persistent RoPE buffers were uninitialized because they are absent from the state dict.
`build_empty_hf_model()` now recomputes `DeepseekV4RotaryEmbedding` buffers from config
after `to_empty`; after that, parity passes. Root-cause evidence:
`handoffs/in_progress/r2_real_weight_layer0_attn_debug.txt` showed embeddings, parameters,
mHC collapsed states, and q-residuals were identical, while RoPE cos/sin diverged before the
fix.

Commands/evidence:

- Layer 0, sliding attention + hash router:
  `handoffs/in_progress/r2_real_weight_layer0_parity.txt`,
  log `handoffs/deepseek-v4/r2_logs/r2_real_weight_layer0_parity_node64.log`.
  Result: PASS, final hidden `rel=0.006770`, `cos=0.99997699`.
- Layer 2, CSA compressor/indexer + hash router:
  `handoffs/in_progress/r2_real_weight_layer2_parity.txt`,
  log `handoffs/deepseek-v4/r2_logs/r2_real_weight_layer2_parity_node64.log`.
  Result: PASS, compressor `rel=0.004233`, final hidden `rel=0.006203`,
  `cos=0.99998093`, router agreement `16/16`.
- Layer 3, HCA + learned top-k router with `seq_len=128`:
  `handoffs/in_progress/r2_real_weight_layer3_parity.txt`,
  log `handoffs/deepseek-v4/r2_logs/r2_real_weight_layer3_parity_node64.log`.
  Result: PASS with bounded learned-router top-k flips: router agreement `123/128`,
  `router_in_flipped rel=0.005582`, `matched_mlp rel=0.008207`,
  final hidden `rel=0.013415`, `cos=0.99991000`.

This is an EP=1 sliced-forward gate. It proves the real native FP8 dequant/key map and
the replicated V4 attention/compressor/mHC/MoE math against HF eager on real weights. R3
now separately proves a single-node EP2 MoE adapter forward/backward through Megatron
`flex/deepep` for learned and hash routers, including Megatron DDP dense/expert buffer
grouping. A tiny full `V4LanguageModel` EP2 DDP smoke also passes; full `PP3_EP8` actor
training remains a later gate.

## Conversion Rules

- Top-level:
  - `embed.weight -> embedding.word_embeddings.weight`
  - `head.weight -> output_layer.weight`
  - `norm.weight -> norm.weight`
  - `hc_head_{fn,base,scale} -> hc_head.hc_{fn,base,scale}`
- Layer HC/norm:
  - `hc_attn_* -> attn_hc.*`
  - `hc_ffn_* -> ffn_hc.*`
  - `attn_norm.weight -> input_layernorm.weight`
  - `ffn_norm.weight -> post_attention_layernorm.weight`
- Attention/compressor/indexer:
  - `attn_sink -> self_attn.sinks`
  - `wq_a/wq_b/wkv/wo_a/wo_b -> q_a_proj/q_b_proj/kv_proj/o_a_proj/o_b_proj`
  - `q_norm -> q_a_norm`, `kv_norm -> kv_norm`
  - compressor `ape/wkv/wgate/norm -> position_bias/kv_proj/gate_proj/kv_norm`
  - indexer paths flatten under `self_attn.compressor.indexer.*`
- MoE:
  - `ffn.gate.weight -> mlp.gate.weight`
  - `ffn.gate.tid2eid -> mlp.gate.tid2eid`
  - `ffn.gate.bias -> mlp.gate.e_score_correction_bias`
  - shared experts `w1/w2/w3 -> gate_proj/down_proj/up_proj`
  - routed experts: `gate_up_proj[e] = cat([w1, w3], dim=0)`, `down_proj[e] = w2`
- FP8 dequant:
  - Any actual FP8 `*.weight` requires sibling actual `*.scale`.
  - Block size is inferred from `weight.shape[-2:] / scale.shape[-2:]`, matching current Transformers finegrained-FP8 dequant.
  - Converter output dtype is bf16 for dequantized frozen base weights unless a caller requests otherwise.
- Rotary buffers:
  - `rotary_emb.*` and nested compressor/indexer `*.rotary_emb.*` buffers are rebuilt from config and are not native checkpoint tensors.
  - The loader strict check allows these missing buffers, matching the existing parity loader's treatment of HF `model.rotary_emb.*`.

## 1-Layer Megatron Slice Gates

Layer 0 command:

```bash
PYTHONPATH=/root/Megatron-LM:$PWD CUDA_VISIBLE_DEVICES=0 \
python -m custom_kernels.deepseek_v4.megatron.slice_torch_dist \
  --checkpoint /nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8 \
  --save /nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8-r2-slice1-torch_dist \
  --num-layers 1 \
  --master-port 29621
```

Result:

- Output path: `/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8-r2-slice1-torch_dist`
- Size: 15G
- Contents: `latest_checkpointed_iteration.txt=release`, `release/.metadata`, `metadata.json`, `common.pt`, `__0_0.distcp`, `__0_1.distcp`
- Native load stats: `direct_tensors=27`, `expert_slices=512`, `missing=()`
- Load-back into a fresh 1-layer model via `dist_checkpointing.load(...)`: `missing=[]`, `unexpected=[]`
- Sample diffs vs native source after load-back:
  - `layers.0.self_attn.q_a_proj.weight`: 0.0
  - `layers.0.mlp.shared_experts.gate_proj.weight`: 0.0
  - `layers.0.mlp.gate.tid2eid`: exact
  - `layers.0.mlp.experts.gate_up_proj[0]`: 0.0

Layer 2 command (loads native source layer 2 into local layer 0 to cover CSA compressor/indexer without a 3-layer materialization):

```bash
PYTHONPATH=/root/Megatron-LM:$PWD CUDA_VISIBLE_DEVICES=0 \
python -m custom_kernels.deepseek_v4.megatron.slice_torch_dist \
  --checkpoint /nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8 \
  --save /nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8-r2-slice-layer2-torch_dist \
  --source-layers 2 \
  --master-port 29625
```

Result:

- Output path: `/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8-r2-slice-layer2-torch_dist`
- Size: 15G
- Native load stats: `source layers: [2] -> local layers: {2: 0}`, `direct_tensors=37`, `expert_slices=512`, `missing=()`
- Load-back into a fresh layer2-shaped 1-layer model: `missing=[]`, `unexpected=[]`
- Sample diffs vs native source after load-back:
  - `layers.0.self_attn.compressor.kv_proj.weight <- layers.2.attn.compressor.wkv.weight`: 0.0
  - `layers.0.self_attn.compressor.indexer.q_b_proj.weight <- layers.2.attn.indexer.wq_b.weight`: 0.0
  - `layers.0.self_attn.compressor.indexer.kv_proj.weight <- layers.2.attn.indexer.compressor.wkv.weight`: 0.0
  - `layers.0.mlp.gate.tid2eid <- layers.2.ffn.gate.tid2eid`: exact
  - `layers.0.mlp.experts.gate_up_proj[0] <- source layer2 expert0`: 0.0

Layer 2 global-key command (builds only native source layer 2 as local `layers.0`, but
preserves checkpoint `ShardedTensor.key` as global `layers.2.*`):

```bash
PYTHONPATH=/root/Megatron-LM:$PWD CUDA_VISIBLE_DEVICES=0 \
python -m custom_kernels.deepseek_v4.megatron.slice_torch_dist \
  --checkpoint /nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8 \
  --save /nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8-r2-slice-layer2-global-torch_dist \
  --source-layers 2 \
  --preserve-global-layer-ids \
  --master-port 29632
```

Result:

- Output path: `/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8-r2-slice-layer2-global-torch_dist`
- Size: 15G
- Native load stats: `source layers: [2] -> local layers: {2: 0}`, `direct_tensors=37`, `expert_slices=512`, `missing=()`
- Load-back into a fresh `layer_ids=[2]` 1-layer model: `missing=[]`, `unexpected=[]`, `load_state_dict_missing=[]`, `load_state_dict_unexpected=[]`
- Sample diffs vs native source after load-back:
  - `layers.0.self_attn.compressor.kv_proj.weight <- layers.2.attn.compressor.wkv.weight`: 0.0
  - `layers.0.self_attn.compressor.indexer.q_b_proj.weight <- layers.2.attn.indexer.wq_b.weight`: 0.0
  - `layers.0.self_attn.compressor.indexer.kv_proj.weight <- layers.2.attn.indexer.compressor.wkv.weight`: 0.0
  - `layers.0.mlp.gate.tid2eid <- layers.2.ffn.gate.tid2eid`: exact
  - `layers.0.mlp.experts.gate_up_proj[0] <- source layer2 expert0`: 0.0

## Logical EP Shard Gate

`slice_torch_dist.py` now accepts `--logical-ep-size` and `--logical-ep-rank` for a
single-rank preflight shard. This does not create a complete multi-rank training checkpoint;
it verifies that one rank can materialize only its local expert shard, save expert-axis
`ShardedTensor` metadata, and load that same shard back.

Command:

```bash
PYTHONPATH=/root/Megatron-LM:$PWD CUDA_VISIBLE_DEVICES=0 \
python -m custom_kernels.deepseek_v4.megatron.slice_torch_dist \
  --checkpoint /nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8 \
  --save /nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8-r2-slice1-ep16-rank7-torch_dist \
  --num-layers 1 \
  --logical-ep-size 16 \
  --logical-ep-rank 7 \
  --master-port 29641
```

Result:

- Output path: `/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8-r2-slice1-ep16-rank7-torch_dist`
- Size: 3.0G
- Native load stats: `direct_tensors=27`, `expert_slices=32`, `missing=()`
- Logical EP shard: rank 7 / size 16, global experts 112..127
- Load-back into a fresh logical EP16 rank7 1-layer model:
  `load_state_dict_missing=[]`, `load_state_dict_unexpected=[]`
- Sample diffs vs native source after load-back:
  - `layers.0.self_attn.q_a_proj.weight`: 0.0
  - local expert 0 <- global expert112 `gate_up_proj`: 0.0
  - local expert 0 <- global expert112 `down_proj`: 0.0
  - local expert 15 <- global expert127 `gate_up_proj`: 0.0
  - local expert 15 <- global expert127 `down_proj`: 0.0

## Real EP2 torchrun Gate

`slice_torch_dist.py` now also supports a true torchrun preflight:

```bash
PYTHONPATH=/root/Megatron-LM:$PWD CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m custom_kernels.deepseek_v4.megatron.slice_torch_dist \
  --checkpoint /nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8 \
  --save /nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8-r2-slice1-real-ep2-v2-torch_dist \
  --num-layers 1 \
  --pp-size 1 \
  --ep-size 2 \
  --master-port 29653
```

Result:

- Output path: `/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8-r2-slice1-real-ep2-v2-torch_dist`
- Size: 15G
- Each rank loaded `direct_tensors=27`, `expert_slices=256`, `missing=()`
- DCP metadata for `layers.0.mlp.experts.gate_up_proj` is global shape
  `[256,4096,4096]` with two chunks at expert offsets 0 and 128.

Load-back command:

```bash
PYTHONPATH=/root/Megatron-LM:$PWD CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m custom_kernels.deepseek_v4.megatron.verify_torch_dist \
  --checkpoint /nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8 \
  --load /nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8-r2-slice1-real-ep2-v2-torch_dist \
  --output handoffs/in_progress/r2_slice_real_ep2_verify.txt \
  --source-layers 0 \
  --pp-size 1 \
  --ep-size 2 \
  --master-port 29654
```

Load-back result:

- Rank0: global experts 0..127, `q_a_proj` diff=0, expert0/127 gate/down diff=0.
- Rank1: global experts 128..255, `q_a_proj` diff=0, expert128/255 gate/down diff=0.
- `load_state_dict_missing=[]`, `load_state_dict_unexpected=[]` on both ranks.

Root cause closed: the first non-v2 EP2 attempt saved successfully but had incorrect expert
metadata (`gate_up_proj` global shape `[128,...]`, one chunk only) because
`LanguageModule.super().sharded_state_dict()` did not call `V4GroupedExperts.sharded_state_dict()`.
`V4LanguageModel.sharded_state_dict()` now explicitly overwrites grouped-expert entries with
global expert-axis `ShardedTensor` metadata. Do not use the non-v2
`...-r2-slice1-real-ep2-torch_dist` artifact; it is superseded by the `-v2-` artifact.

## Real PP2 + EP2 torchrun Gate

Command:

```bash
PYTHONPATH=/root/Megatron-LM:$PWD CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  -m custom_kernels.deepseek_v4.megatron.slice_torch_dist \
  --checkpoint /nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8 \
  --save /nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8-r2-slice2-real-pp2-ep2-torch_dist \
  --num-layers 2 \
  --pp-size 2 \
  --ep-size 2 \
  --plan-first-layers 1 \
  --plan-last-layers 1 \
  --master-port 29655
```

Result:

- Output path: `/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8-r2-slice2-real-pp2-ep2-torch_dist`
- Size: 27G
- PP rank0 covers source layer 0; PP rank1 covers source layer 1.
- EP rank0/1 cover global experts 0..127 / 128..255.
- Rank load stats: pp0 ranks loaded `direct_tensors=22`, `expert_slices=256`; pp1 ranks
  loaded `direct_tensors=26`, `expert_slices=256`; all `missing=()`.
- DCP metadata records layer0 and layer1 expert tensors as global expert shape
  `[256,4096,4096]` with chunks at offsets 0 and 128. Embedding is present on pp0;
  output/norm are present on pp1.

Load-back command:

```bash
PYTHONPATH=/root/Megatron-LM:$PWD CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  -m custom_kernels.deepseek_v4.megatron.verify_torch_dist \
  --checkpoint /nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8 \
  --load /nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8-r2-slice2-real-pp2-ep2-torch_dist \
  --output handoffs/in_progress/r2_slice_real_pp2_ep2_verify.txt \
  --num-layers 2 \
  --pp-size 2 \
  --ep-size 2 \
  --plan-first-layers 1 \
  --plan-last-layers 1 \
  --master-port 29656
```

Load-back result:

- Rank0: pp0/ep0 source layer0 experts 0..127, `q_a_proj` diff=0, expert0/127 gate/down diff=0.
- Rank1: pp0/ep1 source layer0 experts 128..255, `q_a_proj` diff=0, expert128/255 gate/down diff=0.
- Rank2: pp1/ep0 source layer1 experts 0..127, `q_a_proj` diff=0, expert0/127 gate/down diff=0.
- Rank3: pp1/ep1 source layer1 experts 128..255, `q_a_proj` diff=0, expert128/255 gate/down diff=0.
- `load_state_dict_missing=[]`, `load_state_dict_unexpected=[]` on all ranks.

## Real EP8 torchrun Gate

Command:

```bash
PYTHONPATH=/root/Megatron-LM:$PWD CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  -m custom_kernels.deepseek_v4.megatron.slice_torch_dist \
  --checkpoint /nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8 \
  --save /nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8-r2-slice1-real-ep8-torch_dist \
  --num-layers 1 \
  --pp-size 1 \
  --ep-size 8 \
  --master-port 29657
```

Result:

- Output path: `/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8-r2-slice1-real-ep8-torch_dist`
- Size: 15G
- Each rank loaded `direct_tensors=27`, `expert_slices=64`, `missing=()`.
- DCP metadata records `layers.0.mlp.experts.gate_up_proj` as global shape
  `[256,4096,4096]` with 8 chunks of 32 experts at offsets 0,32,...,224.

Load-back result (`handoffs/in_progress/r2_slice_real_ep8_verify.txt`):

- Ranks 0..7 cover global expert ranges 0..31, 32..63, ..., 224..255.
- All ranks have `load_state_dict_missing=[]`, `load_state_dict_unexpected=[]`.
- All ranks have `q_a_proj` diff=0 and first/last local expert gate/down diff=0.

## Full PP1 + EP16 Actor Launcher Template

`scripts/run.r2.v4.convert.pp1_ep16.sh` is the high-EP 16-rank two-actor-node launcher
template retained for reference. It leaves at least one of the four currently available
H20 nodes for rollout, but the real H20 run OOMed for this model; the working current
fallback is `PP3_EP8`. It wraps:

```bash
torchrun --nnodes=2 --nproc_per_node=8 \
  -m custom_kernels.deepseek_v4.megatron.slice_torch_dist \
  --num-layers 43 \
  --pp-size 1 \
  --ep-size 16
```

Required env per actor node:

- `MASTER_ADDR`: rank-0 actor node IP/hostname.
- `NODE_RANK`: this actor node's rank in `[0,2)`.
- Optional: `MASTER_PORT`, `CHECKPOINT`, `SAVE`, `MIN_FREE_GIB`.

Sanity behavior:

- Refuses non-2-node/non-8-GPU-per-node layouts unless the script is edited.
- Refuses non-integer or out-of-range `NODE_RANK`.
- Refuses to overwrite an existing output directory.
- Refuses to run if the local checkpoint path is missing.
- Refuses to run if output filesystem free space is below `MIN_FREE_GIB` (default 500 GiB).
- Sets both `NO_PROXY` and `no_proxy` to cover the current 4-node cluster IPs/hostnames
  plus `MASTER_ADDR`.

The per-node output estimate, summing torchrun ranks 0..7/8..15, is about 367/367 GiB.
This is the highest-EP topology that leaves rollout capacity under the current 4-node limit.

## Full PP2 + EP16 Launcher Template (Conversion-Only)

`scripts/run.r2.v4.convert.pp2_ep16.sh` is retained only as an all-4-node conversion
script. Do not use it for the current training split because it leaves no H20 node for
rollout. It wraps:

```bash
torchrun --nnodes=4 --nproc_per_node=8 \
  -m custom_kernels.deepseek_v4.megatron.slice_torch_dist \
  --num-layers 43 \
  --pp-size 2 \
  --ep-size 16 \
  --plan-first-layers 21 \
  --plan-last-layers 22
```

Required env per node:

- `MASTER_ADDR`: rank-0 node IP/hostname.
- `NODE_RANK`: this node's rank in `[0,4)`.
- Optional: `MASTER_PORT`, `CHECKPOINT`, `SAVE`, `MIN_FREE_GIB`.

Sanity behavior:

- Refuses non-4-node/non-8-GPU-per-node layouts unless the script is edited.
- Refuses non-integer or out-of-range `NODE_RANK`.
- Refuses to overwrite an existing output directory.
- Refuses to run if the local checkpoint path is missing.
- Refuses to run if output filesystem free space is below `MIN_FREE_GIB` (default 300 GiB).
- Sets both `NO_PROXY` and `no_proxy` to cover the current 4-node cluster IPs/hostnames
  plus `MASTER_ADDR`.

This is high-EP and lower payload than PP1_EP16, but it uses every currently available H20
node. It is therefore not the current actor checkpoint target.
The per-node output estimate, summing contiguous torchrun ranks 0..7/8..15/16..23/24..31,
is about 179/179/188/188 GiB. The default 300 GiB gate is therefore a per-node shard-output
gate, not a full shared-filesystem copy of the 529.63 GiB unique converted payload.

## Full PP3 + EP16 Launcher Template

`scripts/run.r2.v4.convert.pp3_ep16.sh` is a conservative 48-rank launcher template for the
selected all-6-node conversion candidate. It wraps:

```bash
torchrun --nnodes=6 --nproc_per_node=8 \
  -m custom_kernels.deepseek_v4.megatron.slice_torch_dist \
  --num-layers 43 \
  --pp-size 3 \
  --ep-size 16 \
  --plan-first-layers 15 \
  --plan-last-layers 14
```

Required env per node:

- `MASTER_ADDR`: rank-0 node IP/hostname.
- `NODE_RANK`: this node's rank in `[0,6)`.
- Optional: `MASTER_PORT`, `CHECKPOINT`, `SAVE`, `MIN_FREE_GIB`.

Sanity behavior:

- Refuses non-6-node/non-8-GPU-per-node layouts unless the script is edited.
- Refuses non-integer or out-of-range `NODE_RANK`.
- Refuses to overwrite an existing output directory.
- Refuses to run if the local checkpoint path is missing.
- Refuses to run if output filesystem free space is below `MIN_FREE_GIB` (default 700 GiB).
- Sets both `NO_PROXY` and `no_proxy` to cover the current 4-node cluster IPs/hostnames
  plus `MASTER_ADDR`; extend `CLUSTER_NO_PROXY` when the later 6-node allocation is known.

PP1_EP16, PP2_EP16, and PP3_EP16 scripts were syntax-checked with `bash -n`; refusal paths
were tested locally for wrong `NNODES`, invalid `NODE_RANK`, and insufficient
`MIN_FREE_GIB` on the current script. PP3 has not been launched because the extra two H20
nodes are not available yet.

## Remaining Gap

Full `PP3_EP8` checkpoint production is complete, but R2 is not the full training gate:

- Full `PP3_EP8` load-back is still a representative tensor-diff check. Real-weight forward
  parity has passed only on sliced EP=1 layers, not on the full distributed `PP3_EP8`
  checkpoint.
- R3 single-node EP2 MoE adapter, tiny full-model, LoRA+Muon, PP2 P2P shape-adapter,
  and full `PP3_EP8` debug-train-only actor smoke now pass. The full smoke evidence is
  `handoffs/deepseek-v4/r2_logs/r3_pp3_ep8_train_smoke_attempt11.log`: 24/24 Ray GPUs,
  checkpoint load, synthetic rollout dump, `actor_train end`, Muon BF16 Newton-Schulz,
  `train/loss=12.953747749328613`, `train/grad_norm=1.9838293331558539`, model-only
  checkpoint save, and Ray job success. The 186G smoke checkpoint was deleted after
  evidence capture; debug dumps remain in `/nfs/FM/csl_v4r3/debug`.
- This still is not an RL gate: sglang serve, routing replay, and LoRA merge-before-sync
  remain future work. The full smoke also does not print a LoRA-only audit; that audit is
  currently covered by the EP2 LoRA+Muon smoke.
- `PP1_EP16` remains the preferred high-EP direction in principle but OOMed on H20 for this
  model; `PP3_EP8` is the working 4-node fallback that leaves node62 for rollout. When two
  more H20 nodes arrive, revisit `PP3_EP16`.

## Checks

- `python -m py_compile custom_kernels/deepseek_v4/__init__.py custom_kernels/deepseek_v4/megatron/mcore_model.py custom_kernels/deepseek_v4/megatron/model_provider.py custom_kernels/deepseek_v4/megatron/native_checkpoint.py custom_kernels/deepseek_v4/megatron/real_weight_parity.py custom_kernels/deepseek_v4/megatron/slice_torch_dist.py custom_kernels/deepseek_v4/megatron/verify_torch_dist.py tests/test_v4_model_provider.py tests/test_v4_native_checkpoint.py`
- `pytest -q tests/test_v4_model_provider.py tests/test_v4_native_checkpoint.py tests/test_megatron_argument_validation.py` (23 tests)
- Real checkpoint audit command above.
- PP/EP metadata plan JSON files above, including the high-EP-first 6-node H20 candidates.
- 1-layer `torch_dist` slice builds + load-back verifications above.
- Logical EP16 rank7 single-rank shard save/load verification above.
- Real 2-rank EP2 layer0 save/load verification above.
- Real 4-rank PP2+EP2 layer0/1 save/load verification above.
- Real 8-rank EP8 layer0 save/load verification above.
- Real-weight EP1 forward parity for layer0/layer2/layer3 above.
- `bash -n scripts/run.r2.v4.convert.pp3_ep16.sh` plus refusal-path sanity checks.
- `bash -n scripts/run.r3.v4.pp3_ep8.train_smoke.sh`
- `pytest -q tests/test_ray_train_placement.py tests/test_v4_attention_fallback.py tests/test_v4_compressor_fallback.py tests/test_v4_model_provider.py tests/test_megatron_argument_validation.py` (20 tests)
