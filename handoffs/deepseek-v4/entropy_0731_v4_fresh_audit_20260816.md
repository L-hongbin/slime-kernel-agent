# DeepSeek-V4-0731 fresh RL entropy 下降审计（2026-08-16）

## 结论先行

首轮审计截至 2026-08-16 11:13 JST，0731 formal lineage 的 rollout step 37 已完成：
`entropy/rollout_mc=0.304493`，`entropy/train=0.299885`；2026-08-17 的 old-model/dtypefix
复验及其与 pre-fix lineage 的对比见下方更新。本审计的结论是：

1. **下降是真实的，但严格说是“动态筛选后在线总体上的策略 sharpening”**。它同时出现在
   SGLang sampled surprisal 和 Megatron 全词表 Shannon entropy 上，不像单侧 logger、reducer
   或训练引擎计算错误；但两者都只覆盖每步动态接受的样本，尚无跨 step 固定 prompt/prefix 探针。
2. **“旧方法已经严格稳定”这个前提不成立。** `entropy-analysis-0722` 证明 predictive mask
   会系统性增熵；切到 DIS 后没有建立任何长期稳定性。旧模型、旧 v3 数据、同一 DIS 目标的
   `04brfumz` 也从 rollout H 0.5466 降到最低 0.3095，`n7jh3rpi` 从 0.6032 降到
   0.2073。历史文档中“稳定约 0.40”的是另一套 LR=1e-4、r16 attention-only LoRA 配方。
3. **当前最强机制是 reward-conditioned TRLOO + global-token reduction，而目标里没有恢复力。**
   `entropy_coef=0`、`kl_coef=0`，DIS 的 `(0.2, 4.0)` 门在当前 run 保留 99.97% 以上 token，
   几乎等价于无 entropy control 的 on-policy policy gradient。长轨迹又按 token 数贡献更多梯度。
4. **2026-08-17 的 old-model × v4 × fresh 复验已否定“0731 模型是下降的必要条件”。**
   旧模型在相同步数内出现同量级 H 降幅；更早的 pre-dtypefix 旧模型短 run 也已呈下降趋势，所以
   dtypefix 同样不是必要条件。模型/precision contract 仍可影响 H0、斜率和动态接受人口，但应从
   必要主因降级为 modifier。这些复验仍不是估计各自 effect size 的纯 A/B。
5. **当前还不能仅凭在线 H 判定功能性 collapse，但风险比 08-16 更高。** old run 的 step 35
   已到 rollout/train H 0.2616/0.2523；其后半段
   response length 和 truncation 转升、raw reward 转降，H 仍继续下降；“更短、更高回报”的良性
   叙事不再足以解释共同趋势。按当前无恢复力目标，没有理由期待它自行稳定，不能让 3000-step
   计划无固定探针地继续外推。
6. **dtype 修复后的确比紧邻的 pre-fix fresh run 降得更快，但不是 dtype 直接改变模型 entropy。**
   共同 step 0--25 的 rollout/train H 斜率约为修复前的 2.3 倍；同窗 grad norm、PPO KL、
   train-rollout MAE 和长度几乎不变。precision 只在生成 response 后进入 evaluator，因而最强解释是
   reward/filter 接受人口与 TRLOO 梯度结构改变；没有 frozen response/train replay 前不能把斜率差
   当作 dtype 的独立因果效应。

## 2026-08-17 更新：换回旧模型后仍以同量级下降

最新 fresh run 是
`formal_dsv4r21_fp4_pp1cp2_12k_prompt_tvm_v4_release_dtypefix_verified_20260816_fresh`，W&B
[`tpcqslzo`](https://wandb.ai/shuailin_chen/slime/runs/tpcqslzo)。截至 2026-08-17 09:10 JST，
step 36 已完整完成，rollout/train H 为 `0.265962/0.254420`。它使用旧
`DeepSeek-V4-Flash` trainer base、旧模型对应的
`DeepSeek-V4-Flash-DSpark` rollout 部署，仍是相同 v4 release parquet、fresh state 和当前
TRLOO + token-DIS 配方。

### 直接结果

| fresh lineage | 指标 | step 0 | step 34 | 0→34 绝对变化 |
|---|---|---:|---:|---:|
| old Flash + v4 | rollout MC H | 0.451196 | 0.285318 | -0.165878 |
| 0731 + v4 | rollout MC H | 0.494103 | 0.323106 | -0.170997 |
| old Flash + v4 | train full-vocab H | 0.451824 | 0.276876 | -0.174947 |
| 0731 + v4 | train full-vocab H | 0.495909 | 0.318534 | -0.177375 |

表中使用共同且较平滑的 step 0--34 窗口；最新 old step 35 又降到 rollout/train
`0.261554/0.252281`，raw reward 0.658、response length 6,226、truncation 12.5%。该单点不应用于
估计模型效应；step 36 小幅回弹到 `0.265962/0.254420`，仍贴近历史 0.25 诊断线，强化了立即固定
探针而不是继续观望的必要性。

Kimi 独立复核又找回同日被替换的 pre-dtypefix fresh run
[`4h151ccd`](https://wandb.ai/shuailin_chen/slime/runs/4h151ccd)：同一 old train/DSpark pair、v4
parquet、launch contract 和 fresh state，rollout H 从 0.460572@0 呈波动下降到 0.356888@26；
train H 从 0.4601@0 到 0.4100@25，中间最低 0.3646@23。它在 rollout 26 后以 rc=1 结束，不能证明
长期斜率，却足以否定“只有 dtypefix 才触发下降”。当前 verified run 同期下降更快，但两次 source/
runtime 和 filter population 未完全配对，不能把速度差归因给 dtypefix。

old run 的 step 0→34 OLS slope 为 rollout/train `-0.005649/-0.006000` nat/step，两条曲线
相关系数 0.9997。最重要的是，old 与 0731 的 train H 差在 step 0 已有 -0.04409，step 34
仍是 -0.04166；两条线的下降量几乎相同。换言之，旧模型不是“后来崩得更快”，而是它在各自
动态接受人口上的 H0 更低，随后同样持续 sharpening。这直接排除了“只有 0731 权重才会触发下降”。

### 对机制的更新

old run 的 step 23--34 中，train/rollout H 仍约以 `-0.0051/-0.0050` nat/step 下降，但同期：

- raw reward 的 OLS slope 已转为 `-0.0103`/step；
- response length 转为 `+49` token/step；
- truncation 转为 `+0.00273`/step；
- step 30--34 的 raw reward/correctness 五步均值为 0.647/0.524，低于 step 15--19 的
  0.869/0.611，而 rollout H 五步均值从 0.359 继续降到 0.282。

因此，08-16 用 aggregate length/reward 共变支持 `global-token weighting × 长度/reward 结构` 的
证据过强。它很可能放大了 0731 run 中长度和 H 的共同下降，但**总体**长度缩短、truncation 减少、
aggregate reward 上升都不是跨模型 sharpening 的必要条件。这里没有排除同一 prompt 组内的
`advantage × response length` 协方差：TRLOO 只保证 response 级 advantage 和为零，global-token
loss 下的有效和是按 token 数加权的，并不为零。两条 fresh lineage 真正共同且未被反例推翻的是：
reward-conditioned TRLOO/global-token policy gradient、`entropy_coef=0`、`kl_coef=0`、几乎不触发
的宽 DIS gate，以及动态 reward/filter selection。**最强但尚待冻结 replay 闭合的机制，是 policy
gradient 的净方向持续 sharpen，而目标没有 entropy/fixed-reference 恢复力。**

简单相关也不能替代因果：old run 中 raw reward 与 H 的水平相关是 -0.517，但一阶差分相关仅
-0.093；low-variance drop count 与 H 一阶差分相关仅 +0.003。动态 filter 仍是重要混杂，原因是
它改变接受人口的组成，而不是每轮 drop 数本身。当前 checkpoint 34 的 dataset state 为
`sample_offset=1342, sample_index=21472`，0731 对照为 `1644, 26304`；即使 parquet、fresh contract
和 nominal seed 相同，两条线上实际消费、接受的 prompt population 也已明显分叉。

filter 规模也很大：当前 verified old run 的 step 0--35 共丢弃 540 个 `reward_std<0.001` group；
pre-dtypefix old run 的 step 0--26 为 576 个；0731 run 为 850 个（其中 step 1/5 的 61/303 是极端
refill 离群点，剔除后常态均值约 14.5/step）。这些计数不能直接换算成固定候选集上的因果贡献，
但足以说明 accepted population 不是小扰动。verified run 的 drop 数反而更少而 H 下降更快，也排除
“必须依靠更高 drop 率才下降”的简单解释。

现有 sign-conditioned diagnostic 还给出一个一致但非因果的信号：step 0→35 中，`train logp >=
rollout logp` token 的 conditional H 从 positive/negative-advantage 下的 0.363/0.389 降到
0.162/0.172；`train logp < rollout logp` token 则仍约为 0.551/0.543。这符合“概率上移集中在已经较尖
的 context、概率下移集中在较高 H context”的自增强图景。但该 split 使用跨引擎 ratio、未保留各组
token fraction，也不是当前更新的 \(dH\)，所以只能作为机制线索，不能替代 DIS 路径的 first-order
entropy direction 或冻结 replay。

### dtype 修复前后：斜率加快，但只能是 selection-mediated

紧邻的 pre-fix lineage 是
`formal_dsv4r21_fp4_pp1cp2_12k_prompt_tvm_v4_release_20260816_fresh`；修复后 lineage 是
`formal_dsv4r21_fp4_pp1cp2_12k_prompt_tvm_v4_release_dtypefix_verified_20260816_fresh`。两者使用
同一 old Flash base、v4 release parquet、fresh state 和训练超参。pre-fix 最后一个 rollout 没有对应
train step，因此 entropy 对齐到共同完成的 step 0--25：

| 指标 | pre-fix | dtypefix |
|---|---:|---:|
| rollout MC H OLS slope | -0.002381 / step | -0.005506 / step |
| train full-vocab H OLS slope | -0.002525 / step | -0.005917 / step |
| rollout H 前 5-step→后 5-step 均值 | 0.4479→0.3946 | 0.4435→0.3360 |
| train H 前 5-step→后 5-step 均值 | 0.4484→0.3927 | 0.4438→0.3277 |

下降加快不是优化器或 mismatch 突然放大的表象。同一窗口的均值为：

| 指标 | pre-fix | dtypefix |
|---|---:|---:|
| grad norm | 0.04026 | 0.04008 |
| PPO sampled-token KL | 0.003204 | 0.003312 |
| train-rollout logprob MAE | 0.03111 | 0.03106 |
| response length | 6457 | 6354 |
| truncation | 0.1137 | 0.1050 |
| raw reward | 0.6707 | 0.6251 |

precision 的代码路径在 response 生成后才构造 KernelGym payload
（`examples/kernel_agent/generate_with_cuda_agent.py:535-572`），所以它不能直接改变同一
checkpoint/prefix 的 logits。两跑 step 0 尚未经历各自的第一次参数更新，rollout H 已为
0.4606/0.4512；这个初始差只能是 sampled/accepted population 差异，而不是 dtype 让权重变尖。

修复真正改变的是 reward 与 filter：共同 step 0--26 的全零 drop 从 576 降到 404；至少一个继承
FP16 的同序首轮 group 从 16/16 全零 drop 变成 accepted。于是每步虽仍训练 16 个 group，具体 prompt
和组内 success/failure 结构已不同。此前全零、无梯度的组若变成“少数成功、多数失败”，TRLOO 会把
正 advantage 集中到少数 winner，再把 response-level advantage 广播到其全部 token；global-token
reduction 使净更新还取决于组内 `advantage × length`。当前又没有 entropy bonus 或 fixed-reference
KL，DIS 几乎不删 token，这种更有选择性的梯度一旦净方向为 sharpen，就没有恢复项阻止后续轨迹
继续分叉。

但“少 drop”本身不是逐步降熵旋钮：dtypefix run 的当步 drop 数与 rollout H 一阶变化相关仅约
0.009。更可能起作用的是 **哪些 group 被接收，以及组内 winner/loser 的 reward、长度和 entropy
结构**。当前日志只保存最终 aggregate 和被 drop 的审计，没有保存所有候选与 accepted 样本的完整
联表；因此 selection-mediated 是最强机制，而 dtype 独立解释了多少斜率差仍未识别。详细 drop、
precision 分层和真实 task 证据见
`handoffs/deepseek-v4/dynamic_filter_zero_reward_analysis_20260816.md`。

### 这不是纯模型 A/B

两条 fresh run 锁住了 v4 parquet、fresh LoRA/optimizer/RNG/cursor、TRLOO、token DIS、采样温度、
LR/Muon、12k context、LoRA 配方和 overlong penalty；所以它足以回答“0731 是否必要”，不足以
估计 model effect：

1. 0731 run 使用 2 个 8-GPU rollout engine，old run 使用 1 个，调度和随机流不同。
2. 异步生成与 refill 没有冻结 response，候选消费不具备逐样本配对性。
3. 两次运行之间加入了 task precision-contract 修复，它会改变 reward/filter population；历史候选
   尚未 paired replay。
4. 两条在线 H 都只测各自 accepted population，没有跨 checkpoint 固定 prompt/prefix probe。

旧 rollout 目录也不是 0731 权重污染。其 `README.md` 明确说明 DSpark 是“same checkpoint with an
additional speculative decoding module”；目录和 payload 时间为 2026-07-15，早于 0731 的 08-01。
与旧 trainer checkpoint 相比，它的 index 多 3,140 个 tensor，全部为 `mtp.*` speculative module。
因此应把它视为旧模型的正常 serving 变体，而不是混合模型 bug；trainer/serving 的 fixed-prefix parity
仍需测量，但当前 MAE 仅 0.0297→0.0365、DIS valid 仍高于 0.99986，没有异常 mismatch 爆炸证据。

## 1. 审计对象与切点

- 原 Codex session：`entropy-analysis-0722`，thread
  `019f8768-e966-7c80-a717-0ef8fe083317`；原始 JSONL：
  `/root/.codex/sessions/2026/07/22/rollout-2026-07-22T10-20-27-019f8768-e966-7c80-a717-0ef8fe083317.jsonl`。
- 当前 fresh run：`formal_dsv4_0731_fp4_pp1cp2_12k_prompt_tvm_v4_release_20260815_fresh`，
  W&B [`wt63rmt7`](https://wandb.ai/shuailin_chen/slime/runs/wt63rmt7)。它在 step 35 后崩溃，
  从 checkpoint 34 恢复为 [`un6qo86u`](https://wandb.ai/shuailin_chen/slime/runs/un6qo86u)，
  因而 step 35 被重新 rollout/train，一条拼接曲线不能把两个 step 35 当成独立连续更新。
- 当前配置源：`scripts/dsv4/run.deepseek_v4_flash.fp4.formal.rl.sh`；fresh base、prompt 数据和状态
  清空契约见 187--226 行，DIS 配置见 156--169 行。
- 当前首段实证日志：`node69_slime:/tmp/dsv4_formal.out.prev.012305`；恢复后日志：
  `node69_slime:/tmp/dsv4_formal.out`。审计只读，没有停止、修改或重启训练。

## 2. 当前曲线：不是单点噪声

下表的 step 0--35 来自同一 `wt63rmt7` 历史；step 36--37 来自 checkpoint 34 恢复后的
`un6qo86u`。恢复后的重复 step 35 为 rollout/train 0.314069/0.312830，与首段
0.317896/0.314556 同量级，说明恢复没有制造新的断崖。

| rollout step | rollout MC H | train full-vocab H | response len | trunc. | raw reward | sampled logp | DIS valid |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.494103 | 0.495909 | 7257 | 0.1875 | 0.477 | -0.4714 | 0.999987 |
| 10 | 0.523956 | 0.526351 | 8611 | 0.4063 | 0.458 | -0.4970 | 0.999958 |
| 20 | 0.404874 | 0.400518 | 7816 | 0.2188 | 0.742 | -0.3939 | 0.999907 |
| 30 | 0.330860 | 0.326717 | 5703 | 0.0469 | 0.897 | -0.3216 | 0.999881 |
| 35, first run | 0.317896 | 0.314556 | 5074 | 0.0703 | 1.144 | -0.2994 | 0.999761 |
| 36, resumed | 0.305832 | 0.303088 | 6020 | 0.1016 | 0.760 | -0.3013 | 0.999744 |
| 37, resumed | 0.304493 | 0.299885 | 5016 | 0.0000 | 1.004 | -0.3011 | 0.999736 |

首段 step 0--35：

- rollout MC H 下降 35.7%，train H 下降 36.6%；两条逐 step 曲线相关系数 0.9998。
- 5-step 均值比端点更稳健：response length 从 step 0--4 的 8170 降到 step 30--34 的
  5408，truncation 从 0.307 降到 0.059；raw reward 从 0.490 升到 0.999。
- sampled logp 变得更不负，说明实际 rollout policy 在被接受轨迹上提高了置信度。
- PPO sampled-token gap 从 0.00270 增到约 0.008，train/rollout logprob MAE 从 0.0299
  增到约 0.044；有增长但没有失控，且不足以解释 rollout 侧自身的同向 H 下降。

因此，“Megatron 单侧 entropy logger 算错”不符合证据。更精确的说法是：**当前 policy 在每步
动态接受的轨迹/前缀上变尖了**；是否在固定问题和固定前缀上也同幅变尖，尚未测量。

## 3. 两个 entropy 指标到底测什么

### `entropy/train`

`slime/utils/ppo_utils.py:719-755` 对 Megatron TP 分片 logits 精确计算

\[
H(p)=-\sum_v p_v\log p_v.
\]

max、exp-sum 和 \(E_p[z]\) 都跨 TP all-reduce。512-token chunking 只分块执行相同全词表
softmax，不改变定义。`slime/backends/megatron_utils/loss.py:1681-1684` 再对 response mask
内 token 聚合。

formal 默认通过 `scripts/dsv4/_dsv4_task_args.sh:62-70` 展开
`--calculate-per-token-loss`；`slime/backends/megatron_utils/cp_utils.py:209-250` 在 DP×CP 汇总后
除以全局有效 token 数。因此当前 `entropy/train` 和 policy loss 都是 **global-token mean**，
不是每条 response 等权。

### `entropy/rollout_mc`

`slime/backends/megatron_utils/data.py:499-511` 对 SGLang 返回的 sampled-token logprob 计算

\[
\widehat H_{MC}=\frac{\sum_{t\in M}-\log p_{rollout}(x_t\mid h_t)}{|M|}.
\]

在固定、未后选择的 on-policy prefix 总体上，\(x_t\sim p\) 时其期望等于 Shannon entropy；
本 run 虽然 T=1、top-p=1、top-k=-1，没有采样截断，但指标是在整条轨迹经过 reward/有效性
动态筛选后才汇总的，因此它**不是原始 prompt 分布上无偏的固定探针**。

两条曲线高度一致的意义是：排除了单一引擎、单一公式的主要伪影；它不能排除二者共享的
accepted-population drift。

## 4. 旧方法并没有建立“稳定 entropy”

### 4.1 predictive-DPPO 的因果结论是增熵，不是稳定控制

历史 run `th36noa0` 的逐步一阶方向、固定 prompt 与冻结 replay A/B 已证明 predictive
mask 是强增熵放大器（冻结 A/B 为 4.70 倍），不是把 H 锁在目标附近的控制器。推导、
阈值扫描和原始证据统一见
`handoffs/deepseek-v4/predictive_entropy_collapse_20260721.md` 的 §8；本审计只使用该结论，
不再复制证据表。

### 4.2 DIS 也没有长期稳定保证

DIS 代码在 `slime/utils/ppo_utils.py:170-258`：

\[
r_t=\exp(\log p_{train}-\log p_{rollout}),\qquad
M_t=1[0.2<r_t<4.0].
\]

它只删除 ratio 越界 token，没有固定 entropy 方向。历史和当前实证如下：

| lineage | 关键条件 | rollout H 轨迹 | 说明 |
|---|---|---|---|
| `qsnifb60` | iter279 恢复，token DIS | step280 0.6908，后段 W&B summary 0.5315 | 只验证 resume/DIS wiring；没有稳定趋势 |
| `04brfumz` | 旧 model + original v3 + 同 DIS，从 iter724 恢复 | 0.5466@725 → 0.3095@788 → 0.3395@802 | 旧数据也会降；DIS valid 0.9674→0.9384，MAE 0.222→0.330 |
| `n7jh3rpi` | 旧 model + level1cap10 新数据，从 iter724 恢复 | 0.6032@725 → 0.2073@828 → 0.2326@838 | 明确持续下降 |
| `wt63rmt7`/`un6qo86u` | 0731 + v4 + fresh | 0.4941@0 → 0.3045@37 | fresh 0731 也持续下降 |
| `4h151ccd` | old Flash + v4 + fresh，pre-dtypefix | 0.4606@0 → 0.3569@26 | 波动下降；rollout 26 后 rc=1 |
| `tpcqslzo` | old Flash + v4 + fresh + dtypefix | 0.4512@0 → 0.2853@34 | 与 0731 同期绝对降幅几乎相同 |

历史 release 文档 `handoffs/deepseek-v4/release/03_lora_rl_training.md:91-108` 的“稳定约 0.40”
对应 LR=1e-4、r16/alpha32 attention-only LoRA；当前是 LR=5e-6、r32/alpha32、rsLoRA、
LoRA+4 和 shared-expert LoRA。该历史记录没有 run ID、持续步数、固定 prompt 曲线或置信区间，
不能当作当前 formal 配方的稳定性证明。

## 5. 根因排序

### 5.1 已确认：目标没有 entropy/reference 恢复力

- `scripts/dsv4/run.t1.deepseek_v4_flash.rl.sh:122-129` 默认 TRLOO、`entropy_coef=0`；
  `slime/utils/arguments.py:1316-1319` 的 `kl_coef` 默认也是 0。
- `slime/backends/megatron_utils/loss.py:1761` 的实际 loss 是
  `pg_loss - entropy_coef * entropy_loss`，所以当前 entropy 只是仪表。
- `slime/backends/megatron_utils/loss.py:989-1002` 显示 `kl_coef=0` 时不计算 fixed-reference
  reward penalty；W&B 的 `train/ppo_kl` 只是 rollout sampled logp 减 train logp，不是 reference KL。
- `examples/kernel_agent/kernel_reward.py:16-71` 的 TRLOO 是组内 reward 中心化后乘
  \(n/(n-1)\)，没有全局 advantage whitening；同一 trajectory 的标量 advantage 被广播到
  response token。

所以只要 reward-conditioned 梯度的净方向是强化少数高回报模式，H 就会持续下降；当前目标没有
“低于某个 H 自动拉回”的机制。

### 5.2 2026-08-17 修正：aggregate 长度缩短不是必要条件，组内 token weighting 仍待测

formal 默认 global-token reduction，意味着一条 10k-token response 的总梯度权重大约是 5k-token
response 的两倍。当前又有 `scripts/dsv4/run.deepseek_v4_flash.fp4.formal.rl.sh:114-120` 配置的
soft overlong penalty：超过 10,240 token 后线性扣分，到 12,288 token 最多扣 0.5；实现见
`examples/kernel_agent/utils.py:698-728`。

合理机制是：长且失败/被罚的轨迹在每个 token 上带负 advantage，被 global-token reduction 放大；
较短、可复用的成功模式持续被强化，于是长度、sampled surprisal 和 H 一起下降。这与 0731 run 的
aggregate 行为一致，但 08-17 old-model run 在后段 mean length/truncation 转升、reward 转降时仍持续
降 H，所以“总体长度不断缩短”不是必要解释。这没有排除组内 reward/advantage 与 response length 的
协方差：response 级 TRLOO advantage 虽按组中心化，global-token loss 下的有效 advantage mass 是
`sum_i A_i * L_i`，不必为零。仍需 **advantage sign × success/failure × within-group length × entropy**
的 sample-level 分解和冻结 replay，才能判断它是主驱动还是放大器。

### 5.3 已排除为当前主因：DIS gate

当前 step 0→37 的 DIS valid 从 0.999987 变到 0.999736，即 step 37 也只删除约 0.0264% token；
低侧 clip 0.0221%，高侧 0.0043%。这个量级不能解释约 0.19 nat 的 H 下降。DIS 低侧截断在旧 run
可能是放大器，但当前段基本是 ordinary importance-weighted TRLOO。

### 5.4 重要混杂：动态 filter 同时改变训练和测量总体

`examples/kernel_agent/kernel_filter.py:56-143` 会丢弃有效样本少于 9 的组，以及 pre-overlong
task reward 标准差低于 0.001 的组。formal 每步最多生成 24 个 prompt groups，只接受 16 个
（launcher 271--278 行）。全失败、全成功及无方差组都可能被排除。

因此随着 policy 改变，哪些 prompt/response 进入 loss、`entropy/train` 和 `entropy/rollout_mc`
也会改变。当前已有真实的低方差 drop 日志，但没有保存所有候选组的 pre-filter entropy，无法定量
拆出“同条件 sharpening”和“accepted mix 变化”的比例。

随后对大量 `reward=0` group 的独立审计见
`handoffs/deepseek-v4/dynamic_filter_zero_reward_analysis_20260816.md`。该审计确认 step 0--18
共丢弃 414 个 16/16 全零 group，并区分了真实模型失败、单轮配置、稀疏 reward 放大和旧 dtype
合同误配。Slime 与 KernelGym 现已支持任务 precision 端到端传递；修复后 matched step 0--26 的
全零 drop 已从 576 降到 404，low-precision drop 从 171 降到 110，并有同序 prompt 被直接 rescue。
但 refills 提前停止后访问人口与训练轨迹分叉，尚无 fixed-response paired replay，所以仍不能把
172 个净减少或 entropy 斜率差全部归因给 dtype。

### 5.5 次级风险：train/rollout mismatch 与 runtime

当前 MAE 约 0.03→0.044、PPO sampled KL 约 0.003→0.008，没有爆炸；而 rollout MC 自身也降，
所以 mismatch 不是充分解释。但 `entropy-analysis-0722` 已证明旧模型固定 prefix 上仍有跨引擎
残差，当前 0731/FP4/DSPARK/LoRA 状态尚未重做 H0 parity gate。08-17 复核还纠正了一处日志归属：
0731 `wt63rmt7` 和当前 `tpcqslzo` 的 args dump 都没有 `SGLANG_BUILD_COMMIT`；显式的
`28b095c...` pin 属于中间的 pre-dtypefix `4h151ccd`。因此 serving runtime provenance 在主对照中
并未锁死，仍是次级混杂。

## 6. 模型和数据确实变了，但不是单变量实验

| 维度 | immediate old formal | 当前 fresh formal |
|---|---|---|
| 初始化 | iter724 LoRA + Muon optimizer + RNG + dataset cursor | immutable base，LoRA/optimizer/RNG/cursor 全空 |
| train base | `DeepSeek-V4-Flash` | `DeepSeek-V4-Flash-0731` |
| rollout base | `DeepSeek-V4-Flash-DSpark` | `DeepSeek-V4-Flash-0731` |
| 数据 | v3 `drkernel_rl_thinking.parquet`，40,307 rows | v4 release，39,636 rows |
| context/objective | 12k，TRLOO + token DIS `(0.2,4.0)` | 相同 |
| runtime | 未固定等价 immutable build | args dump 同样没有 `SGLANG_BUILD_COMMIT` |

模型不是只换权重名：核心 43 层/hidden 4096/256 routed experts/top-6 相同，但 0731 config 新增
`expert_dtype=fp4`、3 个 compression-ratio 项及 DSPARK block/noise/target-layer/Markov 参数；
权重总字节增加 4.55%，shard 46→48。tokenizer JSON 和 tokenizer config hash 相同；chat template
差异只影响未使用的 `reasoning_effort=max` 分支，因此 tokenizer/template 不是本 run 的直接变因。

v4 数据也不是 v3 的小修：

- 28,125 DrKernel、4,861 CSP-DAG、3,713 KernelBook、1,769 CUDA-Agent、1,168 鸥波生成；
- 旧/新仅 372 个 UUID 直接相同；
- prompt 字符长度 p50 4,857→5,523，p90 5,233→6,701，p99 5,716→10,948。

不过按 seed 42 和已保存 dataset cursor 重建的 step 0--34 候选顺序中，各 5-step bucket 的
DrKernel 比例约 70.5%--75.3%，prompt 字符均值约 5,726--5,905，没有“某一步突然进入外部来源段”
这种简单断点。尚未知道的是 filter 后实际 accepted source/length mix。

08-16 时这些变量还完全纠缠，不能说“数据”或“0731 模型”已被单独证明为根因。08-17 的
old-model + v4 + fresh 复验进一步否定了“0731 是必要条件”，但由于 topology、dtypefix 和 accepted
population 仍不同，model/data/fresh 各自效应大小仍未由完整 factorial A/B 识别。

## 7. 最小无污染消歧方案

优先级按信息量/成本排序；遵循仓库约定，先隔离 rollout 和 train，不启动完整新环路。

### P0：先回答是不是固定条件下真降熵

1. 从 base、iter 4/9/14/19/24/29/34/39 取 checkpoint。
2. 固定 128 个分层 prompt（v3 DrKernel、v4 DrKernel intervention、CSP、其他来源），固定
   teacher-forced prefixes 和 token positions。
3. 每个 checkpoint 用 trainer full-vocab forward 记录 H、top-1 mass、top-k mass、logit std；
   另以固定 seed rollout 记录 sampled surprisal、长度、truncation、reward。
4. 这个 probe 必须绕过动态 reward filter。现有 `--entropy-common-probe` 只冻结同一个 training
   batch 的 mask，不是跨 checkpoint 固定 prompt probe，不能替代本实验。

若 fixed-prefix H 也降，确认真实 conditional sharpening；若在线 H 降而 fixed probe 不降，主因是
selection/prompt/length population drift。

### P1：拆动态 filter

用 `--debug-rollout-only --save-debug-rollout-data` 从同一 checkpoint/seed 生成一轮候选；保存全部
24 groups、filter decision 和 pre-penalty reward，而不只保存最终 16 groups。分别计算 all-candidate、
valid-before-low-variance 和 accepted 三个人口的 H/MC-H、source、length、reward、truncation。

### P2：拆 model × data

做只读 fixed-probe 的 2×2：old base × v3、old base × v4、0731 × v3、0731 × v4；所有臂使用
同一固定 prompt packet 和 seed。现有 old + v4 online run 只回答了“0731 是否必要”，不能代替该
paired probe。若要比较训练斜率，再统一 zero-LoRA、optimizer/RNG、token budget，只跑 1--2 个
冻结 replay update，不能拿 iter724 mature resume 与 fresh step0 直接相减。

### P3：闭合优化目标的因果链

从同一 checkpoint、同一 optimizer/RNG clone、同一保存 rollout 开始，用
`--debug-train-only --load-debug-rollout-data` 比较：

1. current DIS；
2. 无 DIS gate 的 direct IS；
3. current objective + fixed-reference KL；
4. current objective + entropy target/regularizer；
5. overlong penalty on/off（filter decision 保持相同）。

每臂记录固定 probe 的真实更新前后 \(\Delta H\)，以及 advantage sign × success/failure × length bin
的 token 数、梯度一阶 \(\Delta H\)。当前 DIS 几乎不触发，所以比继续微调 DIS 阈值更有信息的是
“无恢复项”对“fixed-reference/entropy target”。

### P4：重做 old/0731 H0 parity

在当前 old 与 0731 的 FP4、DSPARK、LoRA checkpoint 上，对完全相同 prefix 比较 SGLang/Megatron
的 sampled-token signed/absolute logprob gap、full-vocab H、TV、top-1 和 routing replay。不能用
在线 MAE 或沿用 0722 旧模型数字当作当前 parity 证明。

## 8. 当前运行建议

- **不建议把在线 H≈0.25 单独当作功能性 collapse 证明**；但 old run 的 reward/correctness 已从
  中段峰值回落而 H 继续降，风险高于 08-16，不能再用“更短、更高回报”作为继续外推的依据。
- **也不应相信它会自行稳定**：当前 objective 没有任何恢复力，旧/0731、fresh/mature、v3/v4 的
  多条 DIS lineage 都曾下降。
- old run 的 iter34 已是完整 checkpoint，不必再等 iter39；应立即对 base、iter4/9/.../34 跑 P0
  固定 probe。把历史 release 的 `<0.25` 诊断线、`<0.20` hard-stop 线暂作安全护栏，而不是跨配方
  定律。若 fixed-prefix H 同步下降、H 接近 0.25 且 reward/正确率不再增加，建议在下一个安全
  checkpoint 暂停；若接近 0.20，应硬停并先做 P3 replay A/B。
- 不应直接恢复 predictive mask：它已被因果 A/B 证明是强增熵放大器，不是可调目标控制器。

## 9. 独立复核与限制

- Codex 三条只读审计分别复查了 session baseline、最新 run/model/data diff 和 entropy/reducer
  数据链；Kimi CLI 又独立解析了 session、代码和 node69 日志，确认当前端点、DIS 惰性和“旧稳定性
  前提不成立”的核心结论，并找回 pre-dtypefix `4h151ccd` 作为必要性反例。
- Kimi 初审提出“global-token flag 未找到”和 `04brfumz` 本地证据缺失；Codex 随后分别通过
  `scripts/dsv4/_dsv4_task_args.sh:62-70` 以及 W&B `scan_history` 闭合。Kimi 的 per-sample reducer
  保留意见不适用于当前 formal 展开参数。
- `kimip` wrapper 与当前 CLI 的 `--prompt`/`--yolo` 组合不兼容；本次改用同一 Kimi CLI 的
  `kimi -p` 只读模式完成复核。没有训练或仓库行为偏差，wrapper 兼容性缺口仍在。
- Kimi 将 node64/node69 日志的 9 小时显示差初判为 clock skew；直接比较 Unix epoch 后确认两机时钟
  同步，只是 node64=`Asia/Tokyo`、node69=`Etc/UTC`。该稳定事实已写入 `RUNTIME.md`。
- 尚未闭合的核心缺口是：fixed-prefix H、pre-filter/all-candidate population 分解、paired model/data
  effect，以及 current objective 对 fixed-reference/entropy target 的冻结 replay A/B。
