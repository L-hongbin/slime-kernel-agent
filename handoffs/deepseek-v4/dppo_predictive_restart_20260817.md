# DeepSeek-V4 predictive-DPPO lineage and CP1 resume（2026-08-17）

更新时间：2026-08-18。本文件记录当时的运行里程碑，不代表当前仍有训练进程存活。

## 结论

旧版 DeepSeek-V4-Flash 已用 predictive-DPPO 从 iteration/rollout/step 0
重新启动，并完成首个端到端训练 step。原始 fresh W&B run 为
[`6j035ear`](https://wandb.ai/shuailin_chen/slime/runs/6j035ear)，该 lineage 后续保存到
iteration 9。2026-08-17 14:17 UTC 已从 iteration 9 用三台机器、单训练
节点 CP1 拓扑 resume；该阶段 W&B run 为
[`svi5x846`](https://wandb.ai/shuailin_chen/slime/runs/svi5x846)。2026-08-18 的
padded-length 降序单变量重试已完成 rollout/step 10 的 32/32 microbatches 和
optimizer update，并继续进入 step 11。该次更新没有形成新的 durable checkpoint；
截至本 handoff，正式进程仍从最新 iteration 9 恢复并重做 rollout 10，W&B run 为
[`c1cq2gu0`](https://wandb.ai/shuailin_chen/slime/runs/c1cq2gu0)。

## 正式配置

- objective：`dppo_topk_kl_predictive`，没有 DIS ratio gate；
- Top-K：20；tail estimator：`aggregated`；KL threshold：0.15；
- sampled-token importance-ratio cap：5；TIS 关闭；rollout log-probs 开启；
- 旧模型 packed iteration 0、prompt_tvm_v4 release、12,288 context、LR 5e-6、
  LoRA r32/alpha32/rsLoRA/LoRA+4、Muon 0.18 均保持不变；
- fresh 阶段使用 16 张训练 GPU（node69 + node64）与 8 张 rollout GPU（node53）；
- fresh root：
  `/nfs/FM/csl_v4r21_fp4_pp1cp2_12k_prompt_tvm_v4_release_dppo_predictive_delta015_ratio_logged_20260817/out`。

## Step 0 验收

- rollout0：256/256；两次 candidate fetch，第一次过滤后为 accepted 4 / missing 12；
- 低方差丢弃：23 个 prompt groups；每条审计记录包含 sample ID、
  `filter_reward` 与原始 `reward`；
- actor train：32/32 microbatches，optimizer step 成功；
- `train/dppo_importance_ratio=0.9999397416`；
- `train/dppo_importance_weight=0.9999397416`；
- `train/dppo_topk_kl=0.0028356961`；
- `train/dppo_outside=0.0002289148`；
- `train/pg_clipfrac=0.0001380221`；
- `entropy/train=0.4527106706`；
- `train/pg_loss=-0.0164874661`；
- `train/ppo_kl=0.0029165408`；
- `train/grad_norm=0.0472292183`。

上述核心指标已从 W&B API 读回。driver、Ray 三节点和 KernelGym 64 workers / 16 GPU
均健康；日志没有 fatal、OOM、Traceback 或 NCCL 错误。

rollout1 期间两个候选 CUDA kernel 在隔离子进程中 segfault，使对应 GPU worker 短暂
进入 `suspect/non-accepting`；KernelGym 自动 top-up 后连续四次恢复为 16/16
healthy/accepting。共享服务未重启，训练未中断。

## 观测补齐与启动注记

predictive-DPPO 原实现使用 sampled-token ratio 但未直接上报它。现在 loss 额外返回
只读的 `dppo_importance_ratio`（cap 前）和 `dppo_importance_weight`（cap 后），经统一
logger 写入文本与 W&B；计算仍使用同一个 capped tensor，目标函数和梯度不变。
相关 predictive/launcher/filter/metric 测试共 70 项通过。

第一次 predictive launch（W&B `dnuano6i`）只进行到模型加载、未开始 rollout 或训练。
发现上述观测缺口后将其停止，保留 8 KiB 空 debug root，并使用新的 ratio-logged
fresh identity 正式启动。此前 DIS lineage 已完整保存 checkpoint 39，没有删除。

## Iteration 9 到 CP1 的 resume 记录

- 源 checkpoint 为 iteration 9：PP1/CP2/EP8、world size 16、32 个 DCP shard；
  adapter、Muon optimizer 和 RNG 均成功恢复，dataset state 为
  `offset=404, epoch=0, group=404, sample_index=6464`，从 rollout 10 继续；
- 目标 trainer 为 node69 单机 8 GPU：PP1/CP1/EP8、world size 8。node64 和
  node70 各运行一个 8-GPU DSpark rollout engine；
- Megatron 的 torch-dist restore 需要对 CP2→CP1 产生的 `io.BytesIO`
  聚合 payload 执行 `torch.load(..., weights_only=False)`；修复后 8/8 adapter
  恢复、optimizer 766 tensors / 459,735,040 bytes 恢复、RNG 恢复全部通过；
- node64 的 `oscar_hy3_test` 曾在首次 CP1 launch 后重新启动 TP8
  SGLang，导致该节点 8 个 rollout scheduler 被杀。该次 W&B `pbaqfu7v`
  没有完成 rollout/update；已可逆地停止该容器，不是删除。
- 该次 run `or65wenz` 已完成 iteration-9 restore 和向两个 rollout
  engine 的首次 LoRA 权重同步；14:20 UTC 的手工日志样本显示
  rollout 10 为 16/256，两个 engine 都在处理真实候选，未见
  fatal/OOM/Traceback/NCCL 错误。此为“已健康启动”证据，尚不是
  rollout 10 训练 update 已成功的证据。

## Padded-length 降序单变量重试（2026-08-18）

`or65wenz` 的 rollout 10 已收齐 256 个训练样本，但旧的 rank-local 执行顺序
仍是原样本序；actor 在 6/32 后 OOM。失败现场为 GPU1 仅余 1.19 GiB，
PyTorch allocated 70.89 GiB、reserved-but-unallocated 18.90 GiB，下一次申请
1.50 GiB 失败。checkpoint 和 dataset cursor 均仍停在 iteration 9 / offset 404，
因此可以准确重做 rollout 10。

新开关 `--sort-train-microbatches-by-padded-length-desc` 只重排已经分配给各
DP rank 的 microbatch，不改变 DP balancing、样本集合、padding 宽度、loss 或
optimizer。BSHD padding 是对 microbatch raw max 的单调向上取整，因此 scheduler
按 raw max 降序与最终 padded-width 降序严格等价。turn-aware 路径当前只允许
每 trajectory 一个训练样本；formal `max_turns=1` 满足该约束，未来真正多 turn
会 fail-close，避免破坏 trajectory-major 顺序。

重试证据：

- 相关调度、actual-length padding、task-arg、launcher、resume-preflight、sync
  共 103 个不重复测试通过，shell syntax 与 `git diff --check` 通过；
- 三节点 runtime manifest 一致：
  `a2654214a4f5889725855e0c9d8b3eec840f5b59806465484fbd9f876f7c85d8`；
- node69 exact resume preflight：native iteration 9、dataset offset 404、
  start rollout 10、load optimizer/RNG、CP1/EP8；
- W&B run `svi5x846`；rollout 10 在 764.36 秒内收齐 256/256，低方差过滤
  16 groups；
- DP0 的真实 padded-width 顺序从 10 个 12,288 开始，随后
  11,264→10,240→…→4,096，日志为
  `microbatch_widths_descending=True`；Ray 去重行同时标记其余 7 ranks 重复，
  证明 8/8 都执行新路径；
- 最重的前 10 个 12,288 microbatches 连续通过，已经严格越过旧的 6/32 OOM
  位置；最终 actor train 32/32，耗时 847.79 秒，optimizer update 成功；
- step 10：`train/pg_loss=-0.0170568323`、`entropy/train=0.4632303453`、
  `train/dppo_importance_ratio=0.9999143929`、
  `train/dppo_importance_weight=0.9999063837`、
  `train/dppo_topk_kl=0.0037102582`、`train/ppo_kl=0.0038324563`、
  `train/grad_norm=0.0352128520`，均有限；其中 entropy、importance ratio/weight、
  Top-K KL 和 grad norm 已从 W&B API 的 step 10 行独立读回；
- allocator 仍为 `expandable_segments:False`，recompute 仍为每层 uniform(1)，
  没有同时启用其他显存缓解项。step 11 随后开始，继续打印降序宽度证据。

结论：对这次真实、且比失败批次 DP0 更重（12,288 bucket 为 10 个）的 batch，
largest-first 足以消除已复现的碎片型 OOM。它不降低单个 12K microbatch 的理论
峰值，因此若以后在第一个最大 bucket 就 OOM，下一优先级应是增加 activation
recompute 粒度（例如每 2 层一个 checkpoint 边界的等价配置需先单独 smoke），
而不是继续调整排序。

同步过程中另发现并修复一个运维缺口：`node64_dspark` 与 node64 source checkout
共享路径，旧脚本在无 rsync 的容器内会用 tar 把目录复制到自身；现在该 target
跳过 self-copy 但仍从容器视角做完整指纹验证，同时同步排除不属于 runtime
manifest 的 `local_artifacts/`。首次失败没有损坏源码，修复后 3/3 节点验证通过。

## node53 + node70 正式恢复（2026-08-18）

按用户当时的指定，拓扑改为 node69 单机 8-GPU trainer，node53 与 node70 各一个
8-GPU rollout engine；node64 不再属于本次 managed session。启动前 24 张目标 GPU
均为空闲。node53/node70 仅有 PID 1 已收割不了的 zombie，不占 GPU 或端口；node69
另有 64 个 `multiprocessing.resource_tracker` 和 64 个 `multiprocessing.spawn`
历史孤儿进程，已按精确 command line 范围终止，没有触及系统服务。

- node69、node53、node70 runtime fingerprint 一致：
  `fb71e2e3911b4e32e3c8972cfca19ce8d0b3c5419d39f63c93a91546924b4544`；
  TileKernels fingerprint 一致：
  `d99bd6fe6120c7197cc237f8bfc0c7e4375c601e0f06dc2358792e5fa371f9f3`；
- exact `--prepare-only` 验证 native iteration 9、start rollout 10、dataset
  `offset=404/group=404/sample_index=6464`、加载 optimizer/RNG、PP1/CP1/EP8、
  predictive-DPPO、无 DIS/TIS、padded-length 降序均正确；
- KernelGym 三入口均返回 64/64 workers、16/16 GPU healthy/accepting；新提交的
  TVM-FFI smoke `formal_resume_tvmffi_smoke_20260818_124055` 为 cache miss，
  `compiled=true`、`correctness=true`；
- managed launch 成功，driver PID 3436531，日志 `node69:/tmp/dsv4_formal.out`，
  W&B run 为 `c1cq2gu0`；Ray 三节点均 active；
- 8/8 trainer rank 已从 iteration 9 恢复 adapter、Muon optimizer
  `766 tensors / 459735040 bytes` 与 RNG；node53/node70 均 SGLang ready、各 8 个
  DP rank 完成首次 LoRA load，随后两台均出现真实 prefill/decode 与 `/generate 200`，
  rollout 10 正在进行；
- 用户指定本次以及后续里程碑用 Kimi review，不使用 Codex review。Kimi 对上述
  启动证据给出 `PASS`，确认没有 OOM、fatal、Traceback、NCCL 或 Ray node failure。

截至本 handoff，只能宣布“iteration 9 健康恢复、rollout 10 正在运行”，尚不能宣布新的训练
update 已完成。下一强门槛是 rollout 10 收齐、iteration 10 的 32/32 actor
microbatches 与 optimizer update 完成、checkpoint 10 落盘，并把新 LoRA 推送到
两台 engine 后继续 rollout 11。日志中的首次空 adapter unload 400、Megatron Bridge
`cache_position` docstring checker 噪音均非阻塞；KernelGym 样本级 `STATUS -> failed`
代表候选 kernel 评测失败，下一里程碑需确认有效样本和 reward 仍在累计。
