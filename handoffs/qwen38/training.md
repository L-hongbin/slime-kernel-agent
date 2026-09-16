# Qwen3.8 训练配置与 checkpoint lineage

Qwen3.8 使用 BF16 actor、FP8 rollout 和 distributed GDN；维护入口默认 MTP3 rollout 与单步 MTP teacher forcing，见[MTP 训练说明](mtp_training.md)。主工作区的维护入口是 [Qwen launcher](../../examples/kernel_agent/run.t1.qwen3.8.27B.fasync.sh)，当前默认两轮、24K/32K context、R27 和四个 TP4 rollout engine。三轮 40K 使用显式参数，历史快照与运行证据由对应 handoff 维护

稳定镜像、输入 checkpoint、节点和服务路径由 [RUNTIME.md](../../RUNTIME.md) 维护；性能与故障证据集中在[训练验证](training_validation.md)。下面的作业记录用于定位历史结果，不代表作业此刻仍在运行

## 已验证的部署配置

| 配置 | 累积 context cap | actor | rollout | recompute | 维护目录 / 历史源码 |
| --- | --- | --- | --- | --- | --- |
| 单轮 DataV4 / 新 reward | 24,576 | TP4/PP2/CP2，16 GPU | 一台 rollout 节点，2×TP4 | 按单轮 run 配置 | `slime-dev-csl-2-qwen38-rl-20260819` |
| 两轮 TRLOO | 24,576 / 32,768 | TP4/PP2/CP2，PP 33/31，16 GPU | 两台 rollout 节点，4×TP4；可显式回到两引擎 | R27 | `slime-dev-csl-2`；历史 tag `archive/qwen38-2turn-trloo-20260911` |
| 三轮 TRLOO | 24,576 / 32,768 / 40,960 | TP4/PP2/CP2，PP 33/31，16 GPU | 一台 rollout 节点，2×TP4 | R31 | `slime-dev-csl-2`；历史 tag `archive/qwen38-3turn-trloo-20260911` |

三种部署都保留 BF16 训练状态。FP8 training 尚无本模型完整验证结果；FP8 rollout 不能作为 TE FP8 train、`fp8-param-gather` 或 checkpoint 兼容性的证明。旧 replicated-GDN 的参数覆盖率与 TP2/PP4 估算不适用于当前 distributed-GDN 路径

## 共同运行合同

- GDN 使用 distributed heads、FlashQLA、逐 sequence zigzag CP 和 sequence parallel。`distributed + SP + PP>1` 使用 rank-ordered P2P，关闭 batched/overlapped P2P
- Sampling 为 medium、temperature 1.0、top-p 1、top-k -1。Predictive-DPPO 使用 sampled token、behavior top-20 与 aggregated tail，delta 0.15、ratio cap 5；不叠加 TIS/MIS/old actor
- Batch 为 16 prompt groups × 16 samples，global batch 256；active prompt groups 上限 16。Train/logprob 动态 token cap 分别为 8,192/16,384；它们不会切短一条超长 sample
- 每个 rollout engine 的 request cap 与 CUDA Graph cap 为 128，matched serving 的 static memory fraction 为 0.80，input-logprob chunk 为 256，以容纳 refit 后长历史重算的临时 logits。是否容得下负载取决于实际 active token 和 GDN state，不能用 128×最大 context 推断已通过容量验证
- 首轮来自 parquet；后续轮使用 [TVM-FFI feedback 模板](../../examples/kernel_agent/prompt_config/multi_turn_tvm_ffi_short.yaml)，复用精确 token history 和一致的 routing key。模型反馈合同见[反馈与诊断](../kernel_agent/feedback.md)
- 多轮使用 TRLOO、`finalize-mode=positive` 和 last-turn filter；长度惩罚作用于 zero-based `turn_idx=1`。三轮部署中第二轮仍使用 32K cap，而不是全局 40K cap
- 可选 `PACK_MULTI_TURN_TRAJECTORIES=1` 合并训练中的重复前缀；配置、等价性合同和验证边界见[轨迹合并](trajectory_packing.md)
- 仅 completed candidate forward 到达输出比较、确认为 value mismatch 且没有 runtime/timeout/decoy 证据时，Qwen launcher 才给予 0.25 partial reward。性能 reward 要求 correctness；coverage-RS/PRS 关闭，连续 correctness-gated coverage 保留，加速比 reward 上限为 2.0
- 权重更新使用 `retract -> flush -> refit -> continue`。中断前后的 response token 都保留真实 behavior logprob/top-k 和 loss mask；最终单值 weight-version 不能表达 per-token version span
- HCA 白名单、CUDA/FLA/SGLang 补丁、模型 manifest、GPU 占用与 KernelGym 健康由 launcher 的预检和 RUNTIME 合同约束

## Checkpoint 保留与恢复

[Rolling checkpoint 工具](../../scripts/archive_rolling_node_local_checkpoint.py)在保存启用时由 launcher 托管。默认 `ROLLING_CHECKPOINT_CLEANUP=1`、`ROLLING_CHECKPOINT_KEEP=2`，每 60 秒检查一次。它按所有 actor 节点的分片布局和发布 marker 保留最新两代结构完整 checkpoint，保护尚未发布的新一代；状态写在实验目录的 `rolling_checkpoint_cleanup/`

清理假定同一固定拓扑、迭代号递增的 lineage。结构检查不能替代恢复前的 manifest/分片验证。新 topology、不同 prompt/reward 协议和不同数据 lineage 分开记录；resume 时核对所选 checkpoint、两端分片、tracker、数据 hash 和实际 launcher dry run

`MIN_CHECKPOINT_FREE_GIB=180` 是每个 actor 节点启动时的 checkpoint 容量门禁。历史上发布失败的 distributed-GDN iteration 0、被授权清理的旧 source half，以及未到首次保存点的作业均不可用作恢复源。文件是否仍存在，以所选节点的实际检查为准

## 留存的 lineage 与证据

| Lineage | 已记录完成范围 | checkpoint / 结果入口 |
| --- | --- | --- |
| 单轮 DataV4 24K，新 reward，T=1.0 | step 0–80，随后停止 | 训练 checkpoint `iter_0000079`，部署路径见 [RUNTIME.md](../../RUNTIME.md)；BF16 HF `Qwen/Qwen3.8-27B-RL-DataV4-Mismatch0p25-CTX24576/step80` |
| 单轮 DataV4 24K，旧 reward，T=1.0 | step 0–40，随后停止 | 训练 checkpoint `iter_0000039`，部署路径见 [RUNTIME.md](../../RUNTIME.md)；历史 manifest 验证总量 489,656,713,582 bytes |
| 单轮 DataV4 16K，T=1.1 | step 0–100 | 原实验目录 `iter_0000099`；与 DataV2 的结果见[统一评测](kernelbench_eval.md) |
| 两轮 TRLOO，四引擎 | 留存记录显示完成并发布 iter79 | 独立 `RetractPR.R2N4E` lineage；历史源码 `3cce5650` |
| 三轮 TRLOO，R31 | 8 月 31 日记录了 fresh 启动及 rollout 0 | 历史源码 `fa9b0e42`；不据此推断之后的训练状态 |

旧两轮、三轮及 retract worktree 已移除，维护入口统一为 `dev_csl2`。三个历史 HEAD 分别由 `archive/qwen38-2turn-trloo-20260911`、`archive/qwen38-3turn-trloo-20260911`、`archive/qwen38-retract-partial-20260911` tag 保留；完整源码、未跟踪文件、Git bundle 和逐文件 hash 保存在[清理归档](../../local_artifacts/worktree_cleanup/20260911_150825/manifest.json)。[两轮真实 token 前缀审计](../../local_artifacts/qwen38/two_turn_prefix_audit_20260828.txt)已迁入主工作区。历史源码不等同于当前默认配置；非 MTP 的旧训练方案需要显式关闭 MTP，并按表设置 context、recompute 和 rollout 拓扑

单轮新 reward 作业为 `raysubmit_WcpFVeacBQakBwap`、W&B `dq6jndgq`，日志尾名 `20260826.110957.log`，实验标签为 `Mismatch0p25.NoPRS.Len4096Pen0p2.DataV4...CTX24576`。最后完整 step80 后没有 step81 update；对应评测证据位于 `local_artifacts/qwen38/newreward_step80_eval_20260827/`

两轮四引擎作业为 `raysubmit_CknwuRRptXARvfCX`，启动日志尾名 `20260830.071207.log`，实验标签为 `RetractPR.R2N4E...CTX32768`。此前的 pybind feedback 污染作业和两引擎 `raysubmit_6vi1djKWLCuiuQAy` 没有产生可恢复 checkpoint

三轮作业为 `raysubmit_ztUq6UdHNx7hTzDm`、[W&B avzyn4ig](https://wandb.ai/shuailin_chen/slime/runs/avzyn4ig)，日志为 `experiments/FAsync.NoSpec.CG128.DPPOPredictive.PP33x31.R31.matched.medium.Temp1.0.TRLOO3TP.C24K-32K-40K.TH.Fb8000.Aff.TVMFFI3T.RetractPR.Mismatch0p25.NoPRS.T2Len4096Pen0p2.DataV4.tvm_ffi.Qwen3.8-27B.BF16Train.FP8Rollout.CTX40960/logs/20260831.225541.log`

## 数据与评测边界

训练默认数据为 `Data/prompt_tvm_v4/release/train.parquet`。行数、hash 与 v4.1 TF32 notice 派生协议由[发版文档](../data/synthesize/training_data_release.md)维护。训练内 reward、KernelBench 单轮结果和多轮 best 指标分别使用各自分母；统一评测数据和不配对的 context/reward bundle 不能用于单变量因果结论
