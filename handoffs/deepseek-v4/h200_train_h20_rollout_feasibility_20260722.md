# H200 单机训练 + 4 台 H20 rollout 可行性评估

## 结论与建议

本文是 2026-07-22 的网络实测与容量估算，尚未在 H200 上验证训练可行性。落实这个拆分方案需要双向可路由的数据通道，或带版本协议的跨域 artifact bridge。当时可达的 SSH 中转实测只有 H200→H20 `1.6 MB/s`、H20→H200 中位数 `4.8 MB/s`，不能直接承载正式 RL。四路并发反而把 H200→H20 聚合带宽降到 `1.035 MiB/s`，所以增加连接数或让 H200 分别推给四台 H20 不能解决问题

每步不应传 154.259 GiB frozen base。base 在 H200 和四台 H20 上各自一次性落盘并校验 manifest；每个 policy version 只让 219.20 MiB LoRA adapter 跨域一次，再由一台 H20 relay 在 rollout 网内分发到四个单机 engine。反向还必须传 rollout batch：当时 routing replay 的 int32 expert IDs 比 adapter 更大，应无损转换为 `uint8` 后压缩。正式配置下反向 payload 预计为 `0.65–0.85 GB/step`

推荐基础设施提供至少 `100 MB/s` 双向端到端有效吞吐，最好是可路由的 10 GbE 级通道。达到这个门槛后，adapter 约 `2.3s`、压缩后的 rollout batch 约 `6.5–8.5s`，都能从分钟级降到秒级。现有 fully-async 流水线下，端到端 cadence 预计由当时中位数 `19.3 min/step` 降到 `9–12 min/step`，约 `1.6–2.1x`。整体加速低于四倍，因为训练 GPU 数从 16 张 H20 减为 8 张 H200，而且新瓶颈会落在 H200 train 与四机 rollout 中较慢的一侧

如果网络/对象存储可以按上述门槛提供，生产化约 `8–12 engineer-days`，另计基础设施排期。如果只能继续使用隔离网络，需要实现双向 artifact queue、压缩、版本与故障恢复，约 `14–20 engineer-days`。仅完成“adapter 跨域一次并在 H20 内 fanout”的局部改造约 `3–5 engineer-days`，但它不解决反向 rollout batch，因此不能单独形成可运行闭环

## 训练与 rollout 数值约束

同一台 H200、同一 checkpoint 上，rollout generation 与完整前缀 scorer 仍不一致，关闭 DSPARK 后也存在，说明更换 trainer 硬件不能消除 train-rollout mismatch

H200 trainer + H20 rollout 上线前必须做固定 prefix 的 H0 parity gate，同时报告 sampled-token signed/absolute log-prob gap、完整词表 entropy/top-1/logit scale、routing replay 完整性，以及 batch/policy version 是否严格匹配。结论和测量细节见 [entropy 主 handoff](predictive_entropy_collapse_20260721.md)、[本评估采用的 session 快照](../../local_artifacts/deepseek-v4/r2_logs/entropy_analysis_0722_h200_split_snapshot_20260722.txt)、[同机分解](../../local_artifacts/deepseek-v4/r2_logs/tim_h200_decomposition_20260722.txt)和 [DSPARK 修复证据](../../local_artifacts/deepseek-v4/r2_logs/tim_h200_dspark_root_fix_20260722.txt)

## 实测网络与真实 payload

测量从 H20 `node64` 到 H200 `gpu03`，使用当时唯一可达的 `h200-gpu03` SSH 中转，关闭 SSH compression，以 `/dev/zero` 或 `/dev/urandom` 产生数据并写入 `/dev/null`，不经过磁盘。测量对象是现有传输路径的应用层吞吐，NIC 标称速率不在测量范围内

| 路径 | 实测 | 对设计的含义 |
|---|---:|---|
| H200→H20，单流不可压缩数据 | `1.6 MB/s` | 219.22 MiB adapter 需要 `143.7s` |
| H20→H200，三次 64 MiB 中位数 | `4.8 MB/s` | packed rollout batch 预计 `2.3–3.0 min` |
| H200→H20，4 路并发共 128 MiB | `1.035 MiB/s` aggregate | 中转是共享瓶颈，不能四机各传一份 |
| H20 node64→node69，不可压缩数据 | `159 MB/s` | 跨域一次、H20 内 fanout 是正确拓扑 |

H200 management NIC 和 H20 bond 都有更高的标称速率，但测试时 gpu03 无法直连 H20 `10.11.2.x`，双方高速数据网也不在同一可路由 fabric。瓶颈位于跨域路由和中转，GPU、NVLink 与 adapter load 均未成为限制。完整原始数字、payload 压缩测量和重测门槛见 [网络实测记录](../../local_artifacts/deepseek-v4/r2_logs/h200_h20_network_benchmark_20260722.txt)

当时 adapter-only hot swap 已经具备可复用的主体：formal launcher 传入 `--sglang-enable-lora` 与 `--use-lora-weight-sync`，每步汇总 766 个 BF16 LoRA tensors，共 `229,867,520 bytes`；rollout engine 将其写成约 `229,967,152 bytes` 的 safetensors 并从本机 tmpfs 热加载。历史 formal 的完整 swap 为 `9.6–17.5s`，记录中的一个样本为 `1.9s`，因此换网后 engine load 不会成为分钟级瓶颈。四台 rollout 必须保持四个独立的 8-GPU 单机 engine，因为现有 tmpfs adapter path 不支持一个 engine 跨多机

adapter 本身不容易压缩：真实 safetensors 用 zlib level 1 只有 `0.7948` 压缩比。更大的反向数据来自 routing replay。真实 256-sample dump 中，`rollout_routed_experts` 为 `[1,580,341, 43, 6]` int32，占 `1,630,911,912 bytes`；ID 全在 `[0,255]`，所以转为 uint8 是精确编码，不改变路由。代表性 32-sample 子集的 `uint8 + zlib1` 只有原 int32 payload 的 `22.65%`。按 formal 每批 `2.4–3.0M` token 外推，routing 从 `2.5–3.1 GB` 降为约 `0.57–0.70 GB`，加上 token、log-prob 和 metadata 后按 `0.65–0.85 GB` 规划。该值由真实 dump 外推，仍需在实现后记录 formal live batch 的 packet bytes、压缩 CPU 时间和网络时间

## 推荐传输架构

```text
                           policy manifest vN
H200 trainer  ── adapter_vN.safetensors ──> H20 relay ──> 4 x single-node H20 engine
      ^                                             |
      |          rollout_packet_vN                  |
      └── uint8 routing + lossless compression <────┘
```

### 有直连数据通道时

优先保留现有单 Ray control plane，因为改动最少，但固定 Ray 端口范围并只开放管理流量；Megatron/NCCL collective 只在八张 H200 内，四个 SGLang/TP collective 分别只在各自 H20 节点内，跨域不建立 GPU collective。H200 rank 0 将 adapter 交给一个 H20 relay，避免把同一 Python tensor dict 作为四个跨域 Ray 参数发送。relay 校验后在 H20 网内 fanout，四个 engine 继续走现有 alternating LoRA load/swap

基础设施验收以最终 transport 对 1 GiB 不可压缩 payload 的双向实测为准，要求持续吞吐均不低于 `100 MB/s`；网卡标称速率不能替代这项测量。如果走对象存储，测试必须包含 upload、publish、四节点 pull 或 relay fanout 的真实路径

### 网络继续隔离时

把 H200 train 与 H20 rollout 作为两个独立服务，通过可被双方访问的对象存储、broker 或受控 relay 交换 immutable artifacts：

- trainer 先写 `adapter_vN.safetensors.partial`，完成后生成包含 `base_manifest_sha256`、`policy_version`、size 和 sha256 的 manifest，再 atomic rename/publish；H20 relay 只拉一次并在内网分发
- 四个 engine 全部成功 load 并返回同一 version 后才发布 ready ack；任一失败时继续保留旧 adapter、重试，不允许部分 engine 进入新 version
- rollout manager 把 routing IDs 精确转换为 uint8，用 zstd/zlib 低级别无损压缩，将 batch version、policy version、schema version、token count、checksum 写入 manifest；H200 解压后恢复 int32，再进入现有 routing replay
- 保留两个 adapter slot 和至少两个 rollout packet slot；所有操作幂等，消费成功后再回收。当时 formal 的 `weight_staleness_steps` 恰为 1，新协议必须维持 `<=1`，缺版本就阻塞，不能静默漂到 2 以上
- 传输与计算流水化：H200 训练 batch k 时，H20 rollout k+1 并上传；H20 使用已确认的 adapter version 生成，不在请求中途切权重。现有慢中转在理想流水下可能被部分隐藏，但四路并发退化和双向大 payload 使其没有生产余量，不能把“恰好被算力遮住”当作正式设计

不建议第一版用 FP8/int8 量化 LoRA 或有损 delta。目标 session 已经证明很小的执行路径差异也会进入 sampled-token probability 与优化 ratio；为节省两分钟再引入新的有损误差不划算。LoRA tensors 每步是稠密更新，稀疏 delta 也没有已测收益。安全的压缩点是整数 routing ID 的 uint8 精确编码和普通无损压缩

## 训练与端到端效率

当时可比 formal canary step 60–64 使用 16 张 H20 train、2 台 H20 rollout。五步 actor train 中位数为 `770.8s`（12.85 min），rollout 中位数为 `1012.5s`（16.87 min），端到端 step 中位数为 `1157.6s`（19.29 min）。新方案同时改变 GPU 型号、train 卡数、context-parallel 拓扑和 rollout 并行度，不能用峰值 FLOPS 直接外推

一台 H200 有 8 张 140 GiB 级 GPU。推荐首先验证 `PP1/CP1/DP8/EP8`：它保留 DP8/EP8，去掉 CP2 通信；当时 H20 `PP1/CP2` 训练后显存使用约 `85.5–88.8 GiB/GPU`，H200 容量很可能足够 12k，但 CP1 的 activation 与碎片峰值尚未实测。若 OOM，再退到 `PP1/CP2/DP4/EP8`。两种拓扑都必须使用匹配的 frozen-base 分片，并验证 adapter、Muon optimizer 与 RNG resume；不能假设 16→8 卡、CP2→CP1 checkpoint 可直接读取

设 H200 单卡对当时 H20 单卡的这套实际 workload 加速比为 `r`，单机八卡 train 时间的一阶模型是 `2/r` 倍当时 16 卡时间：

| 实际单卡加速 `r` | H200 8 卡 train 中位数估计 |
|---:|---:|
| `2.0x` | `12.85 min`，只与当时持平 |
| `2.5x` | `10.28 min` |
| `3.0x` | `8.57 min` |
| `3.5x` | `7.34 min` |
| `4.0x` | `6.43 min` |

规划不应使用 `4x` 乐观端。V4 的 W4A16 unpack、MoE dispatch、attention、recompute 和 Python/host 开销不按 H200 peak tensor-core FLOPS同比缩放。按 `r=2.5–3.5` 及当时 step 波动，预算 `7.3–11.5 min/train step`，中心值约 `8.5–9.6 min`。必须先用保存的 rollout 做 `--debug-train-only --load-debug-rollout-data`，两个候选拓扑各 warmup 3 步、计时至少 5 步后替换该估计

四台 H20 rollout 相比当时两台，理想上限是 `2x`；考虑长样本、dynamic filter/refill、KernelGym 环境和 engine straggler，按 `1.6–1.9x` 规划，即 rollout 中位数约 `8.9–10.6 min`。最终 formal cadence 由 train、rollout 和未被重叠的 transfer 三者最大值决定：

| 条件 | 预计 cadence | 相对当时 19.29 min |
|---|---:|---:|
| 新通道达到 ≥100 MB/s，transfer 为秒级 | `9–12 min` | `1.6–2.1x` |
| 继续用当时 SSH 中转，保守计入双向传输 | `13–17 min` | `1.1–1.5x` |

第二行即使能靠异步暂时隐藏部分传输，也会把网络抖动直接转成 staleness 或 GPU 空等，不适合作为正式方案

## 开发量与上线顺序

| 工作 | 有生产数据通道 | 隔离网络 artifact bridge |
|---|---:|---:|
| H200 container、当时 repo/Megatron patch、154 GiB base 落盘与 fingerprint | 2–3 d | 2–3 d |
| PP1/CP1 train-only 显存/性能 gate，CP2 fallback，resume/topology 转换 | 2–4 d | 2–4 d |
| 四 engine launcher、固定端口/ACL、H20 relay 与一次跨域 fanout | 2–3 d | 3–5 d |
| rollout uint8 packet、压缩与双向 artifact manifest/queue | 1–2 d | 4–6 d |
| H0 parity、version/staleness、失败注入、resume 和长 full-loop smoke | 3–4 d | 4–6 d |
| **合计，允许并行** | **8–12 engineer-days** | **14–20 engineer-days** |

H200 `gpu03` 当时已有 8×H200 和可用容器，但容器内 repo/Megatron revision 与当时工作树不同，且其 `/nfs/FM` 实际是本机 `/ssd`，不与 H20 的同名路径共享。代码、base、adapter resume 与 dataset state 都要显式同步和做 hash preflight。基础设施打通路由、ACL 或提供对象存储的等待时间不包含在 engineer-days 内

上线按以下 gate 执行：

1. 先从双方运行环境实测 1 GiB 双向 ≥100 MB/s；未通过时不做 full loop
2. H200 用保存 replay 分别跑 CP1/CP2 train-only，确定显存、吞吐和 checkpoint contract
3. 四台 H20 跑 rollout-only，验证吞吐接近 `1.6x` 以上且 engine 版本一致
4. 只跑 adapter publish/load 100 次 soak，检查 sha、partial failure、双 slot 与内存泄漏
5. 固定 batch/prefix 做 H0 sampled-token/full-vocab/routing parity；这是 `entropy-analysis-0722` 导出的 correctness gate
6. 最后跑至少 10-step fully-async smoke，要求 weight staleness `<=1`、无 partial version、记录实际 packet bytes/transfer time，并用实测替换本文的效率区间

这份评估没有在 H200 上启动训练，也没有配置新的跨域路由；训练速度仍是由当时 H20 基线和显式单卡加速敏感性模型得到的估计。已经实测并可直接用于决策的是当时跨域应用层吞吐、H20 内部对照、adapter 大小/可压缩性，以及真实 rollout routing payload 的编码收益
