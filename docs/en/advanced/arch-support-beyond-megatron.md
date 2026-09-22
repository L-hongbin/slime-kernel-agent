# Supporting Model Architectures Beyond Megatron-LM

While the Megatron-LM framework is highly efficient for parallel training, it can lack the flexibility to support rapidly evolving model architectures like Qwen3Next. Natively supporting the unique structures of these models, such as Gated-Delta-Net, often requires invasive and time-consuming modifications to Megatron's core codebase.

To accelerate the adoption of these cutting-edge models, slime introduces a more agile approach: **instead of deeply re-engineering Megatron, we directly import and wrap the model's official HuggingFace implementation**, embedding it as a "black-box" module into Megatron's parallel training pipeline.

This document uses Qwen3Next 80B-A3B as an example to illustrate this concept.

## Principle and Core Components

Megatron's model instantiation is a two-step process: first, it generates a "layer specification" (`ModuleSpec`) based on the configuration, and then it instantiates the actual PyTorch modules according to that spec.

slime leverages this mechanism by **hijacking the spec generation stage to replace Megatron's native modules** with an external implementation (in this case, from HuggingFace). This process involves the coordination of three core components:

1.  **Replacing the Megatron Module Spec**
    This is the entry point for our solution. We use a custom function (e.g., `get_qwen3_next_spec`) to modify the standard `ModuleSpec`, swapping out Megatron's native Attention layer with our custom wrapper.
    * **Implementation**: It retrieves the standard Decoder Block Spec, points its `self_attention` field to our custom module, and enables model-specific configurations like `qk_layernorm` as needed.
    * **Corresponding File**: `slime_plugins/models/qwen3_next.py`

2.  **Wrapping the HuggingFace Implementation**
    The spec modified in the previous step now points to a wrapper layer, such as `HuggingfaceAttention`. This layer inherits from Megatron's `MegatronModule`. Its core responsibility is to act as a bridge, handling the data alignment required by parallelism strategies (like sequence parallelism), and then internally calling the native `Qwen3NextAttention` module loaded from HuggingFace.
    * **Corresponding File**: `slime_plugins/models/hf_attention.py`

3.  **Aligning Model Weights**
    Once the model architecture is integrated, we must ensure that the weights can be loaded correctly. slime keeps the HuggingFace-to-Megatron name mapping and tensor transforms next to its checkpoint loader.
    * **Corresponding File**: `slime/backends/megatron_utils/hf_to_megatron/qwen3_next.py`

Through the coordination of these three components, we can successfully run a complex model architecture not natively supported by Megatron—using its HuggingFace implementation as the vehicle—on top of Megatron's parallel framework. This is achieved while fully retaining all key capabilities like model parallelism, MoE acceleration, and pipeline scheduling.

## Qwen Full-Attention TP Gate Compatibility

When `attention_output_gate=True`, `get_qwen3_5_spec` and `get_qwen3_next_spec` automatically replace full-attention `SelfAttention` with slime's `TPGatedSelfAttention`. No extra flag or installed Megatron source modification is needed; linear-attention selection remains unchanged.

Older Megatron versions slice query but not gate when TP exceeds the KV head count. The compatibility module applies the matching head slice using the module's TP group rank; newer versions that already return a rank-local gate pass through unchanged. For Q24 / KV4, TP4 retains 6 query/gate heads per rank and TP8 retains 3. Unrecognized mismatched layouts raise an explicit error.

The module adds no parameters, checkpoint keys, or communication, and preserves autograd through the slice. `tests/test_gated_attention_compat.py` covers old/new layouts, every TP4/TP8 rank, output values, and hidden-state/QKV-weight gradients. These CPU layout simulations do not replace real multi-GPU collective validation.

## Distributed GDN A2A Optimizations

Use the existing options together:

```bash
--qwen-gdn-implementation distributed \
--qwen-gdn-a2a-implementation fused \
--qwen-gdn-cache-thd-permutation
```

Both `native` and `fused` already combine packed sequences into a single collective per direction. With CP>1, each GDN layer issues two forward and two backward A2As, excluding recomputation. The fused path additionally combines head permutation with packing and reuses stream-local send/receive scratch buffers.

Packed CP-to-HP sequence reordering runs inside custom autograd. Its backward directly scatters a bijective permutation instead of using generic `index_select` backward zero-fill/accumulation. Ragged HP-to-CP similarly packs reordered sequence rows directly into send scratch and restores the original order in backward. Returned tensors do not alias communication scratch, and backward does not depend on scratch contents from a previous call. The evenly sharded native path is unchanged; ragged sharding still uses its dedicated fused path.

`--qwen-gdn-cache-thd-permutation` also caches Q/KV boundary validation to avoid repeated per-layer GPU-to-CPU synchronization. The cache belongs to the packed batch and tracks the selected boundary tensors' identities/versions, total length, and CP size. Tensor replacement, normal in-place mutations, or layout changes trigger revalidation; initial validation is never skipped.

`tests/test_distributed_qwen_gdn.py` covers cache invalidation, packed/nonpacked layouts, equal/ragged sharding, and outputs/nonuniform gradients across simulated ranks, including Triton pack/unpack on CUDA when available. Collectives are simulated, not a real multi-GPU NCCL performance validation. Performance comparisons should include forward/backward, representative packed lengths, and the training recomputation configuration.

## Current Limitations

* The HuggingFace replicated wrapper described above does not shard the replaced module itself over TP. This limitation does not apply to the distributed GDN path above.
* **Impact**: In most large-scale MoE models, the parameter count of the Attention layer is relatively small, so this limitation typically has a minimal effect on memory footprint and training throughput.
* **Alternative**: Supported Qwen GDN models can use `--qwen-gdn-implementation distributed`, which installs TP/CP-sharded modules through slime's module specifications.
