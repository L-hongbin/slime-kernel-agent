# 在 Megatron-LM 中快速支持新模型架构

Megatron-LM 框架虽然并行效率高，但在支持日新月异的新模型架构（如 Qwen3Next）时，其灵活性有所欠缺。若要原生支持这些模型的特殊结构（例如 Gated-Delta-Net），往往需要对 Megatron 的核心代码进行侵入性较大、开发周期较长的改造。

为了能快速跟进这些前沿模型，`slime` 提出了一种更敏捷的方案：**与其深度改造 Megatron，不如直接引入并封装模型官方的 HuggingFace 实现**，将其作为一个“黑盒模块”无缝嵌入到 Megatron 的并行训练流程中。

本文以 Qwen3Next 80B-A3B 为例，介绍这一实现思路。

## 实现原理与核心组件

Megatron 的模型实例化分为两步：首先根据配置生成“层规格”（`ModuleSpec`），再依据该规格实例化具体的 PyTorch 模块。

`slime` 正是利用这一机制，在**生成 Spec 的阶段“劫持”并替换掉 Megatron 的原生模块**，从而将外部实现（此处为 HuggingFace 模块）无缝嵌入。这一过程主要涉及三个核心组件的协同：

1.  **替换 Megatron 模块规格 (Spec)**
    这是整个方案的入口。我们通过一个自定义函数（例如 `get_qwen3_next_spec`）来修改标准的 `ModuleSpec`，用我们自己的封装层换掉 Megatron 的原生 Attention 层。
    * **具体操作**：获取标准的 Decoder Block Spec，将其 `self_attention` 字段指向我们的自定义模块，并按需开启 `qk_layernorm` 等模型特有配置。
    * **对应文件**: `slime_plugins/models/qwen3_next.py`

2.  **封装 HuggingFace 实现**
    上一步的 Spec 会指向一个封装层，例如 `HuggingfaceAttention`。它继承了 Megatron 的 `MegatronModule`，核心职责是作为桥梁，处理好并行策略所需的数据对齐（如序列并行），然后在内部直接调用从 HuggingFace 加载的原生 `Qwen3NextAttention` 模块。
    * **对应文件**: `slime_plugins/models/hf_attention.py`

3.  **对齐模型权重**
    模型结构跑通后，还需要确保权重能正确加载。slime 将 HuggingFace 到 Megatron 的名称映射和 tensor 变换直接放在 checkpoint loader 旁。
    * **对应文件**: `slime/backends/megatron_utils/hf_to_megatron/qwen3_next.py`

通过这三层协同，我们成功地将一个 Megatron 原本不支持的复杂模型结构（以其 HuggingFace 实现为载体），运行在了 Megatron 的并行框架之上，并完整保留了模型并行、MoE 加速、流水线调度等全部关键能力。

## Qwen full attention 的 TP gate 兼容

`get_qwen3_5_spec` 和 `get_qwen3_next_spec` 在 `attention_output_gate=True` 时，自动将 full attention 的 `SelfAttention` 替换为 slime 的 `TPGatedSelfAttention`，不需要新增启动参数，也不修改安装的 Megatron 源码。linear attention 的选择不变。

旧版 Megatron 在 TP 大于 KV head 数时只切分 query、不切分 gate。兼容模块按当前模块 TP group 的 rank 对 gate 做同样的 head 切片；新版已经返回 rank-local gate 时直接透传。以 Q24 / KV4 为例：TP4 每个 rank 保留 6 个 query/gate head，TP8 保留 3 个。未知的不匹配形状会明确报错。

该模块不增加参数、checkpoint key 或通信，切片保留 autograd。`tests/test_gated_attention_compat.py` 覆盖旧版/新版布局、TP4/TP8 全部 rank、输出值及 hidden-state/QKV-weight 梯度；这是 CPU 模拟布局测试，不替代真实多 GPU 通信验证。

## Distributed GDN 的 A2A 优化

训练时可以组合使用：

```bash
--qwen-gdn-implementation distributed \
--qwen-gdn-a2a-implementation fused \
--qwen-gdn-cache-thd-permutation
```

`native` 和 `fused` 都已经将 packed 序列合并通信；CP>1 时，每层 GDN 的普通 forward/backward 各执行两次 A2A，不计重计算。`fused` 额外将 head 重排融合进打包，并复用按执行 stream 隔离的收发缓冲区。

packed CP→HP 的序列重排在自定义 autograd 内完成，反向利用一一对应的排列直接 scatter，避免通用 `index_select` backward 的清零和累加。ragged HP→CP 同样直接将序列重排写入发送缓冲区，其反向直接恢复原序列顺序。返回张量不引用通信 scratch，梯度计算也不依赖上次调用留下的 scratch 内容。native 的均匀分片路径不变；ragged 分片仍使用专用 fused 路径。

`--qwen-gdn-cache-thd-permutation` 同时缓存 Q/KV packed 边界校验，减少每层重复的 GPU→CPU 同步。缓存绑定当前 packed batch、实际采用的边界 tensor identity/version、总长度和 CP size；替换边界、正常原地修改或改变布局都会重新校验。该缓存不跳过首次校验，也不全局复用不同 batch 的结果。

回归测试位于 `tests/test_distributed_qwen_gdn.py`，覆盖缓存失效、packed/nonpacked、均匀/非均匀 CP 分片、全部模拟 rank 的输出与非均匀梯度，并在有 CUDA 时执行 Triton 打包/解包。通信由测试替身提供，不代表已完成真实多卡 NCCL 性能验证。性能对比应同时覆盖 forward/backward、实际序列分布和训练使用的 recompute 配置。

## 当前限制

* 前述 HuggingFace replicated 封装路径不对被替换模块自身做张量并行（TP）；这一限制不适用于上面的 distributed GDN 路径。
* **影响**：在大多数大规模 MoE 模型中，Attention 层的参数量占比较小，因此该限制对显存占用和训练吞吐的影响通常有限。
* **替代方案**：支持的 Qwen GDN 可以使用 `--qwen-gdn-implementation distributed`，由 slime 的模块规格接入 TP/CP 分片实现。
