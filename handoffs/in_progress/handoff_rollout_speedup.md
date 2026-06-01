# Rollout 加速 — 总览(hub)

drkernel Qwen3.6-27B 多轮 rollout(KernelBench L1,800-sample eval)加速的**总入口**。
各方向的详细记录见下方归档 handoff;本文件只做导航 + 关键结论。

## 环境 anchor(别重复推导)

- 硬件:8× NVIDIA A800-80GB(SM 8.0 Ampere,BF16 312 TFLOPS / INT8 624 TOPS,**无 FP8 tensor core**);NVSwitch 全互联。
- 模型:Qwen3.6-27B hybrid(16 full-attn + 48 linear-attn/mamba)。sglang 0.5.12.post1。
- workload 特点:多轮 + 每轮 KernelGym 外部编译/benchmark(30 warmup + 50 perf trials)。
- **wall 的真瓶颈 = decode 生成量 + KernelGym 外部评测**,不是 prefill / cache。

## 五条尝试方向

### 1. Prefix cache —— ✅ 已结案:对 wall 无效

→ `handoffs/complete/handoff_cache_drkernel_hybrid.md`

试图通过提高 SGLang prefix-cache 命中给多轮 rollout 加速。**结论:cache hit rate ≠ 推理效率。**
四个 run 命中率横跨 21.5%~53%,但 wall 全在 ~2h、s/it 全 ~8——prefill 是配角,decode + KernelGym 才是
主角。两条路线都试过且都不提速:① TP2 vs TP4(减 engine 数能提命中但 TP4 更慢);② preserve_thinking
(no-norm chat template 修复 thinking 归一化把命中救回 + hicache ratio 调好可再 +8pp)。**缓存不是加速杠杆。**
判优看 score 不看 hit。

### 2. W8A8-INT8 量化 —— ⏳ 进行中:~1.07× wall

→ `handoffs/in_progress/handoff_drkernel_w8a8_rollout.md`(实测 + 复现 recipe)
→ `handoffs/in_progress/handoff_w8a8_speedup_survey.md`(只保留未尝试技术的 ranked 决策矩阵)

BF16 → W8A8-INT8 量化 rollout,目标 ~1.5-2× 加速。实测目前 **MLP-only ~1.07× wall**(纯 sglang
ignore_eos 下 per-token ~1.52×,但 slime 端被多轮/KernelGym 摊薄)。A800 无 FP8 tensor core,只能 INT8。
survey 列了 PD-disaggregation / fully-async rollout 等尚未尝试的方向。**未结案** —— 这是目前最有
希望的加速方向(契合"wall 瓶颈在 decode/吞吐")。

### 3. Hadamard rotation 精度根因 —— ✅ 已结案(支撑量化质量)

→ `handoffs/complete/hadamard_rotation_root_cause.md`

rotated BF16(量化常配 SpinQuant/Hadamard rotation)精度掉的根因:**不是 Hadamard 变换本身**,而是
BF16 处理 Qwen3.5/3.6 `apply_layernorm_1p` RMSNorm `(1 + weight)` scale 在 norm fusion 时的精度损失。
为 w8a8 量化的质量保障服务。

### 4. SpecDec(投机解码)—— ⏳ 进行中:EAGLE 实测 ~1.23× wall 且 score 无损

→ `handoffs/complete/handoff_specdec_drkernel.md`(实测结果 + ranked 调研)

EAGLE 链式已实测 **~1.23× wall(1:58:37→1:36:09),score 0.375→0.381 持平(无损)**,是目前最干净的加速,
已优于 w8a8 的 1.07×。但 `accept_length≈3.23` 已逼近 `num_draft_tokens=4` 天花板。未尽路线(ranked):
**P0 EAGLE tree 化 + NGRAM/suffix(均零训练,后者最贴"多轮复刻代码"workload)**;P2 EAGLE3 + 域内 draft(需训)。

### 5. Reward server 并发(KernelGym worker 数)—— ✅ reward worker 8→16:真加速 ~1.14× wall、score 无损

→ `handoffs/complete/handoff_reward_server_concurrency.md`(实测 + RM 时间对比)

reward worker 8→16(**前提:先修 KernelGym compile-cache bug,见 `KernelGYM-reward-only/bug_report.md`**):
wall **1:36:09→1:24:37(5769→5077s,1.14×)、score 0.381→0.384 持平、编译率 57.6%→56.1% 持平**。decode 量两 run 相同 → 692s wall 下降全来自 RM 评测提速。
天花板有限(2× worker 只换 1.14× wall,decode 仍是大头);未修 bug 时堆 worker 会触发并发竞态、编译大面积失败、score 暴跌,是净亏。

## 补充对比

### A100 vs A800 设备差异 —— ⏳ 进行中:推理效率一致

→ `handoffs/in_progress/handoff_rollout_device_efficiency.md`

两条 2026-05-31 EAGLE/rm16 run 对比:A100 wall **1:47:01** vs A800 **1:50:08**,
wall 差异 2.9%;A100 输出更多 token,按 response token 归一是 **1072 vs 1003 tok/s**,差异约 6.9%。
score 同为 0.375、spec_accept_length 同为 ~3.24、decode p90 throughput 基本相同。结论:**A100 和 A800
在这组 rollout 配置下推理效率一致**。注意两份 run.log 未直接打印 `nvidia-smi` 型号,硬件标签来自目录/环境 anchor。

## 总结论 / 下一步

- **加速走量化(w8a8)+ SpecDec(已 1.23× 无损,还有 tree/NGRAM/EAGLE3 可挖)+ survey 里的方向(PD / fully-async)**;prefix cache 不是杠杆;reward worker 不能盲目堆(超订 .40 会掉 score)。
- 任何加速判优**看 score,不看 cache hit / wall 单点**(wall 被 decode + KernelGym 主导,区分度低;rm16 就是被 wall 数字骗的典型)。
