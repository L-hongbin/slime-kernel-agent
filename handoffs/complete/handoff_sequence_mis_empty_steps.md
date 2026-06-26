# flashinfer GDN logprob 失配 → sequence_mis 空训练步:根因与修复 handoff

> 状态:**根因已定量坐实 + 社区交叉验证(2026-06-25)**。修复 = sglang `--sglang-linear-attn-backend triton`。
> 待办 = 改生产脚本 + 重启正式训练(用户确认后)。

---

## ★ 一句话结论 ★

Qwen3.6-27B 是 **GatedDeltaNet(GDN)混合线性注意力**模型。RL 训练里 train/rollout 每 token logprob
系统性失配,使 `sequence_mis`(bound `[0.999,1.001]` = ±0.1%)reject ~96%、产生全 0 空训练步。

**根因 = sglang rollout 的 `flashinfer` GDN(linear-attn)后端数值不准**:对**真实输入**比 `triton`
后端每层发散 ~2e-3(合成随机张量下两者位级一致),沿 48 个 GDN 层**随机游走式跨层叠加** → per-token
logprob 偏 mean ~0.018(≈生产 0.028)、94% token 越过 ±0.1% bound(≈生产 reject 96.5%)。

**修复:`--sglang-linear-attn-backend flashinfer` → `triton`**(或删行,triton 是 sglang 默认)。
实测 reject **96.5% → 0%**,mean|log_ratio| **0.0284 → 0.0045**,max **9.22 → 1.67**,结构性"自信
token 分歧"完全消失。比 fp32-lm-head / fp32-SSM / 放宽 bound 都强,直接消除 mismatch。

社区已知同类 bug:**sglang #20791**(flashinfer GDN `gated_delta_rule_decode_pretranspose` 的
in-place 状态池别名导致精度退化,triton 不受影响);本 run 用 `extra_buffer` 仍坏,说明 logprob 级
失配比该 issue 的 gsm8k 任务级更敏感(详见"社区交叉验证")。

---

## 修复改动

生产脚本 `examples/kernel_agent/run.t1.qwen3.6.27B.fasync.sh`:

```diff
- --sglang-linear-attn-backend flashinfer
+ --sglang-linear-attn-backend triton        # 或直接删除此行(默认 triton)
```

**权衡 / 注意**:
- triton recurrent decode 可能比 flashinfer 慢,需观察 rollout 吞吐;若慢到不可接受,可关注 sglang
  #20791 关联的 PR #21861(flashinfer GDN 修复)是否合入我们的 sglang 版本,届时再评估切回。
- 本次为根因修复,**fp32-lm-head / fp32-SSM / 放宽 bound 全是改错地方或对症缓解,不需要再叠加**。
  (fp32-lm-head 特性代码仍在 `slime/backends/megatron_utils/fp32_lm_head.py`,默认关闭、无害;是否回退由用户定。)
- 可向 sglang 报本 RL/logprob-级证据,补强 #20791(他们用 gsm8k 发现 no_buffer 坏,我们用 logprob 发现 extra_buffer 也坏)。

---

## 根因机制(因果链已闭合)

```
flashinfer GDN decode/prefill kernel 对真实输入比 triton 发散 ~2e-3/层
  (合成 randn 下位级一致;真实 regime 敏感,疑 exp(g)+外置 l2norm vs log-g+内置 l2norm 在真实幅度下 bf16 舍入分叉)
        │  社区 #20791:源头 = flashinfer pool API(initial_state+initial_state_indices)原地读写状态 → slot 复用别名
        ▼
沿 48 个 GDN 线性注意力层经残差流【随机游走式】跨层叠加(√48 × 2e-3 ≈ 0.014)
        ▼
per-token decode logprob 偏 mean ≈ 0.018~0.039(两独立测法 bracket 生产 0.028)
        ▼
94% token 越过 sequence_mis ±0.1% bound  ≈  生产 reject 96.5%  →  空训练步
```

**两次独立确认**(互为验证,均把生产 0.028 夹在中间):
1. 本人:整模型同 prompt greedy,flashinfer vs triton,在**一致前缀 token**(同 id 同位置 = 合法
   teacher-forced 对照)比 per-token decode logprob → mean **0.0175** / max 0.052 / **94.4% 越阈**。
2. codex:固定 2048-token 真实序列双后端 teacher-forced 前向,逐层抓 INPUT hidden state →
   fi-vs-tri 发散 rms **沿深度单调增长**(L0=0 → L20=1.4e-3 → L32=4.5e-3 → L63=**0.084**,全注意力层跳变最大);
   rescore logprob mean **0.0386** / max 4.78。生产序列 ~10k token(远长于 2048),随机游走走得更远 → 略高自洽。

关键澄清:该失配在 **prefill teacher-forced rescore 里也复现**(不是 decode 专有)——任何穿过 48 个 GDN
层的真实输入前向都会叠加。GDN kernel 对真实输入分歧,与 prefill/decode 路径无关。

---

## 决定性证据

**(A) triton 对照(2 节点隔离,rollout_id=0,两侧 base 权重 = 纯引擎失配零 staleness)**

| | flashinfer(baseline) | **triton** |
| --- | --- | --- |
| reject_rate | 0.9647 (246/255) | **0.0000 (0/253)** |
| effective_sequences | 9 | **253(全部)** |
| mean\|lr\| | 0.0284 | **0.00453**(↓6×) |
| max\|lr\| | 9.22 | **1.67**(↓5.5×) |
| 结构性"自信 token 分歧" | 有(缩进 token sglang 99.5% / megatron 0.3%,5.85 nats) | **消失**(worst=` int` 1.7 nats) |

逻辑闭环:megatron-chunkwise(flashqla)≈ fla ≈ **sglang-triton-recurrent**,**唯独 flashinfer 偏离**。
→ 排除 EAGLE(两次都开 EAGLE,triton 就好)、chunkwise-vs-recurrent 固有差异(triton 也 recurrent 却一致)、训练后端。

**(B) 合成 kernel 全路径位级一致(反证"per-step kernel 数学错")**

`repro_flashinfer_gdn_mismatch.py`(同输入喂 flashinfer vs triton vs fp32 ref):decode max≈2e-4、
长序列 N=2000 每步 `flash_vs_triton=0.000`、prefill(T=1024)max=4.9e-4、MTP committed-state max=1.2e-7。
→ 孤立 kernel 数学**等价**;失配只在**真实输入 regime**下出现(下条)。

**(C) 真实输入捕获 replay(`capture_gdn_real_inputs.py`,patched sglang hook,layer=4 真实 A_log regime)**

`flashinfer.decode vs triton.decode`(同真实 q/k/v)output max=**3.2e-3**(合成是 0.000)→ 真实输入下确实分叉。
真实 gating 与合成天差地别:A_log max=+4.94(合成 U(-4,-1))、decay=exp(A_log) 可达 139、**39.5% head
alpha>0.99 / 14.7% alpha>0.999**(近乎完美记忆 head,合成根本打不到)。

**(D) 社区交叉验证** — 见下节。

---

## 社区交叉验证

- **sglang #20791** [Bug][GDN]:flashinfer `gated_delta_rule_decode_pretranspose` 精度退化,triton 不受影响。
  机制 = flashinfer pool API(`initial_state + initial_state_indices`)**原地读写** SSM 状态,slot 复用时
  **状态别名/顺序错乱**;triton 走显式 gather/scatter(每请求拷进拷出)故不受影响。
  gsm8k:`no_buffer`+disable-radix=0.890 / `no_buffer`=0.940 / **`extra_buffer`=0.990(号称不受影响)** / triton 全 0.990。
  状态 Open,关联 PR #21861。**↳ 这正是我们 per-layer ~2e-3 真实输入发散的源头解释。**
- **本 run 的扩展发现**:我们用 `extra_buffer`(#20791 称"不受影响"),但 logprob 级仍 reject 96.5%。
  原因:**logprob 失配(MIS 要 ±0.1%)比 gsm8k 任务准确率敏感几个量级**——对 RL,extra_buffer 不够,必须 triton。
- **sglang #22472**:CuTeDSL GDN decode 在 Qwen3.5 出乱码、triton 正常 → 佐证 triton 是安全后端。
- **sglang #18590**:Qwen3.5/Qwen3-Next GDN 优化 tracking。
- 其它栈同样脆:ollama #15865(GDN recurrent state 写 bf16 → ~0.4%/step 累积)、vLLM #38643 / #25760。

---

## 已穷尽排除项

| 假设 | 证否方式 | 结论 |
| --- | --- | --- |
| fp32 lm-head | train+rollout 双侧 fp32 lm-head,reject 仍 0.984 | 改错层 |
| fp32 SSM 状态 | 查代码:两侧 SSM 本就 fp32(config `mamba_ssm_dtype=float32`) | 空跑 |
| conv1d / decay-gate dtype | 两侧 conv=bf16、decay=fp32,完全一致 | 无 dtype 旋钮可调 |
| megatron 后端 fla vs flashqla | 重跑 train phase:reject/mean/worst-K 几乎一致 | 不是 flashqla 的锅 |
| EAGLE 投机解码 | flashinfer/triton 两次都开 EAGLE,triton 就好 | 排除 |
| chunkwise vs recurrent 固有差异 | triton 也 recurrent,却高度一致 | 排除 |
| 孤立 kernel 单步数学错 | 合成全路径 ≤5e-4(B) | 排除 |
| 层内多步状态累积 | 真实输入 threaded N=1000 饱和 ~2e-3(delta-rule 收缩性封顶),比生产低 14× | 证否 |
| cuda-graph 索引损坏 | 0.24 state 误差在 eager(无 cuda graph)也在 | 排除 |
| GQA broadcast 分叉 | 三 kernel `i_h=i_hv//(HV//H)` 完全一致 | 排除 |
| commit/state 布局错位 | `mamba_state_scatter_triton.py` 按 batch 位索引 = flashinfer `[:batch_size]` | 代码层证否 |
| position 对齐 bug(off-by-one) | codex review:repo 读路径一致 | 不存在 |

---

## 次要工程问题:sequence_mis 空训练步硬化(triton 后已非紧急)

即便 mismatch 修好,`sequence_mis` 是 **hard mask**,reject 高时(如未来 staleness 大)仍可能产生全 0 空步:
训练侧后处理把成形 train batch 全部 mask 置零,但仍提交 optimizer step,污染 step 计数,且
`loss/pg_loss/entropy/ppo_kl/grad_norm` 全 0。这是**训练侧后处理**问题,不是调度层 batch 不够
(`fully_async_rollout` 只保证收够 dynamic-filter 后的 group;MIS 后有效样本/token 数不固定,可降到 0)。

关键代码链:
| 代码 | 行为 |
| --- | --- |
| `examples/kernel_agent/fully_async_rollout.py:316-383` | 收够 `rollout_batch_size` 个 dynamic-filter group 才返回 |
| `slime/ray/rollout.py:831-836` | train step 按 `global_batch_size` 固定切分 |
| `slime/backends/megatron_utils/actor.py:528-545` | 先算 advantage,再 `rollout_data_postprocess`,最后 train(无回 rollout 补样本机制) |
| `examples/kernel_agent/kernel_filter.py:sequence_mis` | 只改 `loss_masks` 置零,样本留 batch 内 → 可全 reject 空步 |
| `slime/backends/megatron_utils/loss.py:1178-1233` | `--calculate-per-token-loss` 全 mask 分子=0,`clamp_min(...,1)` 防分母 0,不报错 |

**已做**:`sequence_mis` 加了 mismatch + effective-sequence/token 统计,进日志 + wandb/TB(`rollout/mis_*`:
`mis_reject_rate` / `mis_mean_abs_log_ratio` / `mis_max_abs_log_ratio` / `effective_sequences/tokens` /
`mis_tok_frac_abs_log_ratio_gt_0.02/0.05`)。单测 `slime/tests/test_sequence_mis_consistency.py`。
(注:逐 token worst-K mismatch dump 调试探针 `SLIME_DEBUG_SEQUENCE_MIS` 及其单测已于 2026-06-26 移除,见下方"已删";`mis_*` 统计指标保留。)

**可选硬化(triton 后 reject→0 已非紧急)**:
- P0:`effective_token_count==0` 时跳过 optimizer/lr step(止血 silent no-op);
- P1:`reject_rate` 设 hard cap,超阈降级为 bounded TIS / clip(改 loss 权重但不减有效样本),替代 hard mask。

**旁注修复**(与主线无关,已改,下次启动起效):
- `rollout/spec_accept_rate`、`rollout/prefix_cache_hit_rate` 恒 0 → kernel-agent 自定义 generate 从没更新
  meta_info。已修(`generate_with_cuda_agent.py:_sample_for_turn` 调 `spec_info.add` / `prefix_cache_info.add`)。
- `train/mtp_loss=nan`:日志除零(reject 高 → 空 microbatch),梯度路径不除、有限,不伤训练,暂不修。

---

## signed bias 旁注

body 的系统性 ~0.5% 偏置(`ratio_mean≈0.995`,`ratio_max` 一直 ≤0.9999)本质 = **−KL(rollout‖train)**:
token 由 sglang 采样、megatron 评分,Gibbs 不等式保证 `E[train_lp − rollout_lp] = −KL ≤ 0`,所以永远是
下界 0.999 在拒,且训练中策略变尖 → KL 增大 → bias 和 tail 同步增大(max 16→22)。这解释了为什么
flashinfer 下"为何系统性偏一个方向";triton 把两套分布拉近后,KL 量级骤降,reject 归零。

---

## 工具与工件

**保留脚本**(根因已闭环,只留 1 个自包含再验证器 + MIS 统计单测;`examples/kernel_agent/test/` 除非另注):
- `evidence_logprob.py` — **唯一保留的 GDN 验证器**:整模型 flashinfer-vs-triton per-token logprob diff(自包含,
  无需 patched hook)。以后验证任何 flashinfer 修复(PR #21861 / 新 sglang 版本)重跑此脚本即可,确认后再从 triton 切回。
- `slime/tests/test_sequence_mis_consistency.py` — MIS 统计一致性单测。

**已删**(根因闭环 + #20791 已存档后清理,结论全在本 handoff;探索代码不再保留):
- sequence_mis 调试探针(2026-06-26 移除):`kernel_filter.py` 的 `SLIME_DEBUG_SEQUENCE_MIS` / `SLIME_DEBUG_SEQUENCE_MIS_TOPK`
  worst-K mismatch token dump 仪表、`slime/tests/test_sequence_mis_debug.py` 单测、`../run.debug.mis_capture.sh` 2 节点隔离编排。
  (生产用的 `mis_*` 统计指标与 `test_sequence_mis_consistency.py` 不受影响,保留。)
- GDN 探索:`repro_flashinfer_gdn_mismatch.py`(合成≡基线)、`capture_gdn_real_inputs.py`(需 patched sglang hook)、
  `cross_layer_hidden_diff.py`(需 qwen3_5.py hook)、`capture_crosslayer.py`、`evidence_gen_flip.py`(英↔中翻转刀刃效应)、
  `replay_multistep_accum.py`(层内累积已证否)、`run_gdn_capture.sh`。
- fp32-lm-head 死路探针:`probe_lm_head_dtype.py`、`probe_fp32_lm_head_tp.py`、`score_same_tokens_sglang.py`。

**数据工件**(`/ms/FM/chenshuailin/mis_debug/`,per-node-local,reboot 前注意转存):
- `rollout_0.pt`(flashinfer rollout)、`worst_tokens_rollout0.jsonl`(max 9.22)vs `worst_tokens_triton.jsonl`(max 1.67)
- `lp_{flashinfer,triton}.json`、`xlayer_lp_{fi,tri}.pt`、`gdn_real_capture_v3.pt`(306MB)

**复核命令**:
```bash
# triton vs flashinfer 对照(隔离 debug)
rg "sequence_mis\] rollout_id=0" experiments/.../logs/*.log   # reject_rate / mean / max
python slime/tests/test_sequence_mis_consistency.py
# 验证修复(整模型 logprob diff,需 2×27B 加载)
python examples/kernel_agent/test/evidence_logprob.py
```

---

## 下一步

1. 生产脚本切 `--sglang-linear-attn-backend triton`,跑一次 sanity(确认 reject ~0、rollout 吞吐可接受)。
2. 重启正式训练(用户确认后)。
3. (可选)向 sglang #20791 补 RL/logprob-级证据;评估 PR #21861 是否值得 cherry-pick 以保 flashinfer 速度。
