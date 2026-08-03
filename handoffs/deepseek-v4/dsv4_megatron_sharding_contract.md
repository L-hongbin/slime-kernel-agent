# DeepSeek-V4-Flash 当前 Megatron 分片契约

## 结论

截至 2026-07-20，r21 正式训练使用 2 个节点、16 张 H20，拓扑固定为：

```text
TP1 × PP1 × CP2 × DP8，叠加 EP8 × ETP1
world_size = TP × PP × CP × DP = 16
```

`EP8` 复用 `DP×CP` 的 rank 空间，不再乘进 `world_size`。当前并行工作的分工是：

- `DP8` 将 global batch 分给 8 个数据副本；每个副本由一个 CP 二元组共同处理。
- `CP2` 将每条序列连续切成两半；每个 rank 仍计算完整的 64 个 query heads。
- `EP8` 将 256 个 routed experts 切成每 rank 32 个；同一 expert shard 在两台训练节点上各有一份副本。

除 routed experts 外，embedding、全部 43 层 attention、compressor、mHC、router、shared expert、final norm 和 lm head 都不做模型权重分片。`TP=PP=1`，因此没有 TP collective，也没有 PP stage 或 PP P2P。旧文档中关于 PP2/PP3 层切分、跨 PP 传输 4 倍 hc-stream，以及“EP-only、每个 rank 重复计算完整序列”的描述均不再代表当前训练。

本文只约束 Megatron 训练侧。当前 rollout 是 3 个独立的 8-GPU DSpark/SGLang engine；SGLang 的 `tp_size=8, dp_size=8, enable_dp_attention=True` 不等于训练侧 TP8，不能直接套用本文的 Megatron rank 轴。

## 1. 当前物理布局与 rank 组

正式入口是 `scripts/dsv4/launch_formal_managed.sh`，它调用 `scripts/dsv4/run.deepseek_v4_flash.fp4.formal.rl.sh`。训练 actor 按 IP、再按 GPU id 排序，所以逻辑 rank 与物理节点的映射为：

| 节点 | IP | GPU | global rank |
|---|---|---:|---:|
| node64 | `10.11.2.164` | 0–7 | 0–7 |
| node69 | `10.11.2.169` | 0–7 | 8–15 |

Megatron 使用 rank 顺序 `tp-cp-ep-dp-pp`。在当前尺寸下，各 process group 为：

| 组 | 当前成员 | 作用 |
|---|---|---|
| CP | `[0,1]`, `[2,3]`, …, `[14,15]` | 一条样本的两个连续序列分片；均为节点内通信 |
| DP（不含 CP） | `[0,2,4,6,8,10,12,14]`；`[1,3,5,7,9,11,13,15]` | 分别固定 `cp_rank=0/1` 的 8 个数据副本 |
| DP×CP | `[0,1,…,15]` | 非 expert 可训练参数的梯度归约、loss/metric 汇总和 checkpoint 副本域 |
| EP | `[0,1,…,7]`；`[8,9,…,15]` | 256 个 experts 的两套 8 路切分；两组都在单节点内 |
| expert-DP | `[0,8]`, `[1,9]`, …, `[7,15]` | 同一 expert shard 的两份跨节点副本 |
| TP / PP / ETP | 每组只有一个 rank | 当前均无通信 |

因此 `EP8` 不是额外占用 8 倍 GPU。它把 16 个 rank 解释为两组 EP8；每个 `ep_rank=e` 持有 expert `[32e, 32e+31]`，并在 global rank `e` 与 `e+8` 上各复制一次：

| `ep_rank` / 物理 GPU | node64 rank | node69 rank | 本地 experts | `cp_rank` |
|---:|---:|---:|---:|---:|
| 0 | 0 | 8 | 0–31 | 0 |
| 1 | 1 | 9 | 32–63 | 1 |
| 2 | 2 | 10 | 64–95 | 0 |
| 3 | 3 | 11 | 96–127 | 1 |
| 4 | 4 | 12 | 128–159 | 0 |
| 5 | 5 | 13 | 160–191 | 1 |
| 6 | 6 | 14 | 192–223 | 0 |
| 7 | 7 | 15 | 224–255 | 1 |

这个布局有两个直接结果：MoE dispatch/combine 留在节点内 NVLink；同一 expert shard 的副本关系和普通 LoRA 梯度归约则跨 node64/node69。

## 2. 权重与模块驻留契约

模型关键尺寸为 `H=4096`、`Dh=512`、`Nh=64`、`q_lora_rank=1024`、`o_lora_rank=1024`、`o_groups=8`、`hc_mult=4`、`E=256`、`I=2048`、`V=129280`。配置有 `base_model_ep_plan`，没有 `base_model_tp_plan`；model provider 也会直接拒绝 `tensor_model_parallel_size != 1`。

“复制”在这里表示参数完整驻留，不表示 16 个 rank 计算相同 token。CP2 下每个 rank 只处理自己的序列半段。

| 部件 | 当前驻留/分片 | 当前计算与通信 | LoRA 状态 |
|---|---|---|---|
| `embed_tokens.weight [V,H]` | 16 rank 全量复制 | 本地 lookup；PP1 下每个 rank 都是首 stage | 冻结 |
| attention 的 `q_a/q_b/kv/o_a/o_b`、norm、sinks | 16 rank 全量复制；attention TP=1 | 全 64 query heads，本地 query 序列；CP 通信见 §3 | `q_a/q_b/kv/o_b` 启用；`o_a` 排除 |
| CSA/HCA compressor 与 indexer | 16 rank 全量复制 | 本地 halo 上压缩；compressed KV 在 CP 组收集 | `kv_proj/gate_proj` 启用；其余冻结 |
| 每层 `attn_hc/ffn_hc` 与 `hc_head` | 16 rank 全量复制；mHC 参数保持 fp32 | 只混合本 rank 的 `[B,S/2,hc,H]`；无 mHC collective | 冻结 |
| TopK/Hash router | 16 rank 全量复制，输出全部 256 个 expert 分数 | 对本 rank token 本地路由，再进入 EP dispatch | 冻结 |
| routed experts | 沿全局 expert 轴做 EP8；每 rank 32 个 | EP 组内 Flex/DeepEP dispatch；ETP1 | 冻结、无 LoRA |
| shared expert MLP | 16 rank 全量复制 | 对本 rank token 本地计算；不参与 EP 分片 | 当前启用 LoRA |
| final norm、`lm_head.weight [V,H]` | 16 rank 全量复制；`tie_word_embeddings=False` | 本地完整 vocab logits；TP1 无 vocab gather | 冻结 |

### 2.1 Attention 不能按 TP 切

V4 使用单 KV head 的 MQA：`num_key_value_heads=1`，再将 KV 广播给 64 个 query heads。CSA/HCA 分支也依赖这一完整的 shared-KV 语义。对 `q_b_proj` 或 query heads 做普通 column TP 会破坏 rank-local query heads 与单 KV head 的匹配，因此当前契约要求所有 attention projection 都保持完整：

- `q_a_proj [1024,4096]`
- `q_b_proj [32768,1024]`
- `kv_proj [512,4096]`
- `o_a_proj [8192,4096]`
- `o_b_proj [4096,8192]`

`o_a_proj` 不是普通 dense linear。它将权重解释为 `[8,1024,4096]` 的分组块对角 BMM；通用 LoRA 会把它误当成一个 `[8192,4096]` dense matrix，改变数学语义，所以当前显式排除 `*.self_attn.o_a_proj`。

### 2.2 Routed expert 的本地存储

每个 rank 的逻辑 expert 权重为：

- `gate_up_proj [32,4096,4096]`
- `down_proj [32,4096,2048]`

r21 直接驻留 official checkpoint 的 packed-MXFP4 字节，不创建对应的 bf16 trainable Parameter：

- `gate_up_proj_fp4 [32,4096,2048]`，scale `[32,4096,128]`
- `down_proj_fp4 [32,4096,1024]`，scale `[32,4096,64]`

dispatch 后按本地 expert 临时解包为 bf16 做 W4A16 计算；固定为一次只保留一个
expert 的临时权重，不再提供大块 grouped transient 开关。自定义 expert 路径保留
DS-V4 的 `clamp + SwiGLU`，不能换成缺少 clamp 的 stock `GroupedMLP` 激活。

## 3. CP2 连续序列分片契约

当前必须使用 `--cp-partition-mode contiguous`，而不是 slime 默认的 zigzag CP。对 padding 后长度 `S`：

```text
cp_rank 0: [0, S/2)
cp_rank 1: [S/2, S)
```

padding 粒度会提升到 `lcm(data_pad_size, CP×128)`；CP2 下即 256 的倍数，保证本地长度 `S/2` 是 128 的倍数。正式 16k 上下文时，每个 rank 的常规本地长度是 8192。

每层 attention 的 CP 数据流如下：

1. 每个 rank 对本地 hidden 计算完整 64 heads 的 query。
2. `cp_rank=1` 从左邻居接收最后 128 个 hidden rows，组成 `[left_halo || local]`；`cp_rank=0` 没有 halo。反向把 halo 梯度送回所有者并累加。raw KV projection 和 compressor 必须共享同一个带 autograd 的 halo exchange。
3. 在 haloed hidden 上计算 raw KV 和 compressor。为去掉重复窗口，CSA（压缩率 4）丢弃 32 个 halo windows，HCA（压缩率 128）丢弃 1 个；每个 rank 最终只保留自己拥有的 compressed windows。
4. CP 组 all-gather 两边拥有的 compressed KV，得到全局 compressed 轴；反向以 reduce-scatter 将所有 query 分片产生的 `dk_comp` 求和后交还窗口所有者。
5. A1 attention kernel 接收本地 query、`q_pos0`、`raw_halo`、局部 raw KV 和全局 compressed KV，直接输出本地 query 半段，不做全序列 query/output all-gather。

attention 之外的 mHC、router、shared expert 和 lm head 都继续在本地序列分片上计算。logprob/token offset、loss mask 和输出重组使用同一套 contiguous offset 规则；训练指标最终在 DP×CP 的 16-rank 组上归约。

当前 CP 路径只支持 dense-over-compressed attention。`V4_SPARSE_ATTENTION=1` 在 `CP>1` 时会硬报错，因为 indexer top-k 尚未实现全局 compressed 轴语义，不能静默退化为 rank-local top-k。

### 3.1 生产采用与验收边界

2026-07-19 实际落地的生产拓扑是 **PP1×CP2×DP8×EP8**，不是早期设计稿中的 PP2×CP2×DP4。PP1 packed base 避开了当时不支持的 PP2→PP1 uint8 distributed-checkpoint 重分片，同时让两个 CP rank 都持有完整 43 层 pipeline，并保持 16-GPU 训练规模。iter59 迁移边界允许一次性的 PP-rank RNG 不连续；LoRA 与 Muon 状态均保留。

正式配置的有界 canary 在 16k 上完成 rollout 60–64，退出码为 0；五步 train time 分别为 770.8、829.1、770.3、865.3、767.5 秒。iter64 在 node69 与 node64 原子提交：每份 checkpoint 都有 32 个 distcp shard、37 个文件、1,154,007,655 bytes，排序后的逐文件 manifest digest 同为 `6e7a4e124928696da13c3198482bd715b3d40d8578722c807ce38067d85d376e`。head-node dataset cursor 为 group/offset 2816、sample 45056、epoch 0；相对 iter59 的额外消耗来自 dynamic-filter refill。

随后，正式 native PREPARE_ONLY 将 iter64 正确解析为 rollout65，并启用 `LOAD_OPTIM=1`、`LOAD_RNG=1` 与 scheduler-horizon override；两份 train-node checkpoint 和 head-node dataset state 均通过检查。正式训练在全部 16 ranks 原生恢复 iter64，并完成 loss/gradient 均为 finite 的 step65。可复核记录见 `local_artifacts/deepseek-v4/r2_logs/r21_pp1cp2_canary60_64_result_20260719.txt` 与 `local_artifacts/deepseek-v4/r2_logs/r21_formal_launch_iter64_20260719.txt`。

## 4. EP8 MoE 执行契约

每层 router 都在本地对全部 256 experts 打分，top-k 为 6：前 3 层使用 `tid2eid[input_ids]` 的 Hash router，其余 40 层使用自定义 `sqrtsoftplus + e_score_correction_bias` TopK router。两种 router 都是完整复制，不按 EP 切权重。

`V4MoELayer` 使用 `moe_token_dispatcher_type=flex` 和 DeepEP backend：

1. 将本 rank 的 token、top-k expert ids 和权重整理成 routing map。
2. 在本节点的 EP8 组内 dispatch，使 token 到达持有目标 expert 的 rank。
3. 按本地 32 个 experts 的连续段执行 W4A16 expert 计算。
4. 在同一 EP8 组内 combine，将 expert 输出送回 token 的来源 rank，并按路由权重合并。
5. 本地计算完整 shared expert，并与 routed 输出相加。

ETP=1，所以 expert 内部没有 column/row tensor parallel，也没有 ETP all-gather 或 reduce-scatter。routed expert 与 router 当前全部冻结，因此没有 expert 梯度归约；`expert-DP=[e,e+8]` 目前主要用于 checkpoint shard 的副本标记。若未来训练 routed experts，它们必须设置 `allreduce=False` 并只在对应 expert-DP 二元组归约，不能进入普通 DP×CP 的 16-rank 梯度桶。

## 5. 当前稳态通信

| 通信 | process group | 当前范围 | 是否存在 |
|---|---|---|---|
| 左侧 128-token halo forward/backward P2P | CP | 相邻 rank 二元组，节点内 | 是，每个 attention 层 |
| compressed KV all-gather / grad reduce-scatter | CP | 相邻 rank 二元组，节点内 | 是 |
| MoE token dispatch/combine | EP Flex/DeepEP | rank 0–7 或 8–15，节点内 | 是 |
| LoRA 参数梯度 all-reduce（由 Muon 更新） | DP×CP | 全部 16 rank，跨节点 | 是；fp32 累加/归约 |
| loss、logprob、metric 汇总 | CP 或 DP×CP | 先按需求恢复 CP 序列，再做全局汇总 | 是 |
| expert 参数梯度归约 | expert-DP | `[e,e+8]` | 当前无，experts 冻结 |
| TP collective | TP1 | 单 rank | 无 |
| PP activation P2P | PP1 | 单 rank | 无 |

因此，旧结论“稳态只有 MoE all-to-all、DP all-reduce 和 PP P2P”已经失效：当前新增 CP halo 与 compressed-KV collectives，同时 PP P2P 已消失。

## 6. 当前 LoRA 与优化器边界

r21 的训练参数为 rank 32、alpha 32、rsLoRA、LoRA+ `lambda=4`。rsLoRA 的 forward scale 为 `32/sqrt(32)`；LoRA+ 让 B 矩阵学习率为 A 矩阵的 4 倍。当前 target set 是：

- 每层 attention：`q_a_proj`、`q_b_proj`、`kv_proj`、`o_b_proj`
- 每个 compressor：`kv_proj`、`gate_proj`
- 43 层 shared expert：`gate_proj`、`up_proj`、`down_proj`

显式排除 `o_a_proj`、router、routed experts、sinks、position bias、mHC 参数和 lm head。训练 MTP 未启用；rollout 的 DSpark draft tree 不改变训练侧 target set。

当前 native checkpoint preflight 审计到 766 个 LoRA tensors、114,933,760 个 LoRA 参数。`LoRA.__call__` 先冻结整个 base，Muon 只拥有这些 LoRA 参数及其 FP32 master/momentum；所有可训练参数都走普通 DP×CP 梯度域，不存在 trainable expert 参数。

## 7. Checkpoint 与拓扑耦合

当前 packed base 是：

```text
/nfs/FM/chenshuailin/checkpoints/deepseek-ai/
  DeepSeek-V4-Flash-FP4-r21-pp1-ep8-torch_dist
```

它必须同时存在于 node64 和 node69 的本地 `/nfs/FM`。转换时 routed experts 按全局 expert 轴切成 EP8，并用 expert-DP rank 标记两份副本；其余权重在 PP1/TP1 下作为复制 shard 加载。

正式训练保存 adapter-only model + Muon + RNG，不重复保存 frozen packed backbone。当前 topology metadata 必须是：

```text
world_size=16
tensor_model_parallel_size=1
pipeline_model_parallel_size=1
context_parallel_size=2
cp_partition_mode=contiguous
expert_model_parallel_size=8
```

native checkpoint 预期有 32 个 distcp shards。由于 `/nfs` 是逐节点本地存储，而 DCP metadata 引用全局 storage set，每个训练节点的 checkpoint 目录都必须包含完整的 32-shard 集合，不能只留下该节点运行时产生的半套文件。

PP、EP、world size 或节点集合变化都必须先生成/验证 topology-matched base 和 resume checkpoint。尤其不能让 `torch_dist` 将旧 PP2 packed-uint8 expert checkpoint 直接 reshard 到 PP1；该路径已经触发过 `Invalid access pattern`。更换训练节点还会改变 IP-sorted global rank，从而改变 checkpoint shard 与物理 GPU 的对应关系。

## 8. 不可静默改变的约束

- `TP` 必须保持 1；model provider 会拒绝其他值。
- 当前正式 `PP` 必须保持 1；43 层和首尾模块都在每个 rank。PP2/PP3 仅是历史/变体拓扑，不得沿用其 checkpoint 或层切分参数。
- `CP2` 必须使用 contiguous 分片，且全局 padding 后长度必须是 256 的倍数。
- `EP8` 必须整除 256 experts；改变 EP 会改变本地 expert 范围和 checkpoint shard。
- `ETP` 必须保持 1；当前 expert 权重没有 expert-internal TP 布局。
- `CP>1` 时不得启用 sparse attention/indexer top-k。
- 正式启动只能通过 `scripts/dsv4/launch_formal_managed.sh`；该脚本和 formal launcher 中烘焙的 topology 才是运行时权威值。

## 9. 可复核依据

- 拓扑与节点：`scripts/dsv4/run.deepseek_v4_flash.fp4.formal.rl.sh`
- Megatron 参数：`scripts/dsv4/full_loop_smoke.sh`
- rank 顺序：`slime/backends/megatron_utils/initialize.py`
- 物理 rank 排序：`slime/ray/placement_group.py`
- 模型、EP dispatcher、checkpoint 分片：`custom_kernels/deepseek_v4/megatron/mcore_model.py`
- CP halo / compressed collectives：`custom_kernels/deepseek_v4/megatron/cp_utils.py`、`attention.py`
- 数据、logprob 与 loss 的 contiguous CP：`slime/backends/megatron_utils/cp_utils.py`
- LoRA target：`custom_kernels/deepseek_v4/megatron/lora.py`
- 已完成的 CP2 canary：`local_artifacts/deepseek-v4/r2_logs/r21_pp1cp2_canary60_64_result_20260719.txt`
- 正式 native resume 证据：`local_artifacts/deepseek-v4/r2_logs/r21_formal_launch_iter64_20260719.txt`

2026-07-20 对 `/tmp/r21_formal.out` 的只读核对确认了 `world_size=16`、`PP1/EP8/TP1/CP2`、node64 ranks 0–7、node69 ranks 8–15；运行中的训练已经越过 canary，本文记录的是其稳定 topology 契约，不承担动态 step 状态记录。
