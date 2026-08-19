# DeepSeek-V4 LoRA adapter serving contract

训练侧只同步可训练 LoRA 参数，rollout 侧保持量化基座不变并热切换 adapter。这条路径避免每步把 LoRA 合入基座后重新量化，也避免 adapter-only 同步创建完整权重更新通信组。本文只定义 serving 与同步不变量，训练目标和超参数由正式 launcher 负责

## 工程结论

- 正式 FP4 rollout 使用 DP8 replicated attention，`SGLANG_ENABLE_DP_ATTENTION=1`、`SGLANG_DP_SIZE=8`、`USE_SGLANG_DEEPEP=0`
- LoRA 要求 attention TP 为 1，不要求 DeepEP；当前 FP4 W4A16 runner 只支持 `a2a=none`
- CUDA Graph 保持开启，正式路径固定使用 Triton LoRA backend
- DSPARK 使用 target-only LoRA，draft worker 不加载 adapter，完整约束见 `handoffs/deepseek-v4/mtp_speculative_decoding.md`
- 当前热切换通过 `slime_lora_0`、`slime_lora_1` 交替实现，切换窗口需要至少两个 adapter slot
- replicated shared expert LoRA 已支持；routed expert LoRA 未支持，并在同步前硬拒绝

## 启用与拓扑约束

正式入口 `scripts/dsv4/run.deepseek_v4_flash.fp4.formal.rl.sh` 经 `scripts/dsv4/run.t1.deepseek_v4_flash.rl.sh` 向 `scripts/dsv4/_dsv4_launch_core.sh` 传递以下配置

| 配置 | 正式值或约束 | 作用 |
|---|---|---|
| `--sglang-enable-lora` | 开启 | 为 rollout engine 注册 LoRA 模块 |
| `--use-lora-weight-sync` | 开启 | 每步只同步 adapter |
| `--lora-dim`、`--lora-alpha` | 与训练配置一致 | 定义 adapter rank 与缩放 |
| `SGLANG_ENABLE_DP_ATTENTION` | `1` | 使 attention 在每个 GPU 上复制 |
| `SGLANG_DP_SIZE` | `8` | 正式单 engine 的 DP 大小 |
| `USE_SGLANG_DEEPEP` | `0` | FP4 W4A16 使用 `a2a=none` |
| `SGLANG_SPECULATIVE_ALGORITHM` | `DSPARK` | 启用 target-only speculative decoding |
| `SGLANG_LORA_BACKEND` | `triton` | 已验证的 CUDA Graph 路径 |
| `SGLANG_MAX_LORAS_PER_BATCH` | `>=2` | 容纳切换时的新旧 adapter |

launcher 在 adapter sync 开启但 serving 未开启，或 slot 数小于 2 时直接失败。`SGLANG_OPT_FUSE_WQA_WKV=0` 保证 `wq_a` 与 `wkv` 仍是独立 LoRA target。所有动态 load、unload 请求广播到各 scheduler，使每个 DP attention replica 的 LoRA pool 同步更新

## 模块映射

| 训练侧模块 | serving 侧模块 | 处理方式 |
|---|---|---|
| `q_a_proj` | `wq_a` | replicated linear |
| `q_b_proj` | `wq_b` | attention TP1 下按 replicated linear 包装 |
| `kv_proj` | `wkv` | replicated linear |
| `o_b_proj` | `wo_b` | attention TP1 下按 replicated linear 包装 |
| `compressor.kv_proj`、`compressor.gate_proj` | `compressor.wkv_gate` | load 时按 kv、gate 顺序融合，并在 scoring 路径显式加 delta |
| `shared_experts.gate_proj`、`shared_experts.up_proj` | `shared_experts.gate_up_proj` | A 侧共享，B 侧按输出维拼接 |
| `shared_experts.down_proj` | `shared_experts.down_proj` | replicated shared expert linear |

以下模块不进入 serving adapter

- `o_a_proj` 是 grouped block-diagonal linear，通用 LoRA wrapper 不适用
- `.indexer.` 子树不是训练目标，必须整体排除，避免 compressor 叶子同名导致尺寸错误和内存破坏
- routed `.experts.` 或 `FusedMoE` 不支持 adapter-only 同步
- MTP draft worker 保持 LoRA-free

shared-expert LoRA 活跃时，`down_proj` 不走绕过 module forward 的 DeepGEMM 路径，确保 adapter delta 生效。compressor 的 `compute_kv_score` 直接读取融合权重，因此必须显式调用 LoRA delta，而不能依赖标准 module forward

## 同步与切换协议

1. 暂停 generation，并 flush 所有在途请求
2. 从各 PP source 收集 `requires_grad` adapter tensor；global rank 0 生成 PEFT state dict 与 config
3. 拒绝 routed-expert 参数，并校验 PP payload 完整性与 adapter rank 一致性
4. 每个 engine actor 在唯一 tmpfs 目录落盘 `adapter_model.safetensors` 与 `adapter_config.json`
5. 所有 worker 从持久路径加载新的交替名称，全部成功后才发布 active name 与 `weight_version`
6. 恢复 generation，后置卸载旧名称；卸载失败记录为 pending，并在下次切换前重试
7. 每个 `/generate` 请求携带当前 active `lora_path`

多 worker 共享内存 tensor transport 存在 backing file 提前释放竞态，因此当前协议使用 tmpfs 路径，并把目录保留到 unload 或安全复用。新 adapter 加载失败时旧 adapter 继续驻留，active name 与版本号均不前进

交替名称是当前不可变 adapter API 的兼容方案。固定名称、固定 slot 的原位 copy 在结构上可行，但尚未实现对应 endpoint，也未完成 graphs-on 多步验证；在此之前不得用 unload 后同名 reload 替代交替方案

## CUDA Graph 与执行不变量

- 常规 CUDA Graph 保持开启；仅 piecewise 或 `torch.compile` graph 会因 LoRA 自动关闭
- idle DP rank 的 `lora_ids` 可能为空列表，eager idle 与 graph replay 都必须把静态 batch state 刷新为全 `None`
- `prepare_lora_batch` 必须显式接收 `use_cuda_graph` 目标，不能从 IDLE mode 猜测写入哪组 buffer
- graph padding 后的 segment 与 weight-index 尾部必须清零，禁止复用上一 active bucket 的状态
- compressor `wkv_gate` LoRA 活跃时禁用该层的 multi-stream overlap，使 batch metadata 与 LoRA kernel 位于同一 stream
- `.indexer.` 排除必须发生在 LoRA wrapper 注册阶段，不能只在 adapter load 时过滤
- `lora_enabled` 按 runner 设置，DSPARK target runner 开启，draft runner 关闭
- Triton 是正式 backend；CSGMV reload 和 chunked idle graph 路径不在支持范围

## 已闭合的故障模式

| 故障 | 根因 | 固化修复 |
|---|---|---|
| 多 worker 加载偶发缺文件 | 共享内存 tensor backing file 被先完成的 worker 删除 | 使用唯一且保留到 unload 的 tmpfs 目录 |
| wrapper 初始化 `KeyError` | `__getattr__` 代理了 wrapper 自己注册的属性 | 从代理中排除 `weight`、`bias`、`base_layer` |
| compressor graph replay illegal access | alternate stream 读取 default stream 准备的 batch metadata | adapter 活跃时在主 stream 执行 compressor delta |
| idle rank replay 旧 adapter 状态 | 空 `lora_ids` 跳过 graph static state 刷新 | eager 与 replay 都准备全 `None` idle state |
| decode 中移动位置 illegal access | indexer compressor 被叶子名误匹配并复用尺寸不同的 pool | 在 wrapper 注册时排除完整 `.indexer.` 子树 |
| shared-expert adapter 无效 | optimized path 绕过 `down_proj` module forward | adapter 活跃时关闭对应 bypass |

这些是实现不变量，不应保留逐次 smoke 编号、机器状态或临时调试过程。环境专属事实归 `RUNTIME.md`，实验结果归对应 handoff

## 验证与支持边界

`test_sglang_lora_wrapper_attr_proxy` 必须在尚未加载 train-side TileLang 的干净解释器中先运行，避免 `libcudart_stub.so` 遮蔽真实 CUDA runtime。其余契约随后在另一个解释器中运行

```bash
python -m pytest \
  tests/deepseek-v4/test_dsv4_lora_serve.py::test_sglang_lora_wrapper_attr_proxy -q

python -m pytest \
  tests/deepseek-v4/test_dsv4_lora_serve.py \
  tests/deepseek-v4/test_dsv4_lora_shared_expert.py \
  tests/deepseek-v4/test_dsv4_mtp_lora_phase1.py \
  -q -k 'not test_sglang_lora_wrapper_attr_proxy'
```

GPU 验证使用 `scripts/dsv4/diagnostics/lora/lora_reload_repro.sh` 隔离 rollout 侧，必须覆盖 CUDA Graph、idle DP rank、连续 decode、至少三次 adapter swap 和失败回退。零 adapter 应与 frozen base 一致；非零 adapter 的 rollout logprob 应跟随训练策略，且不随 LoRA norm 增长产生重新量化型漂移

SGLang 的可复现实现由 `scripts/dsv4/patches/dspark_port_series/0004-*`、`0007-*`、`0008-*`、`0012-*`、`0013-*` 承载。当前不支持 routed-expert LoRA、固定 slot 原位更新、CSGMV reload、chunked backend idle graph，以及交替双名称之外的多 adapter 并发 serving
