# DeepSeek-V4 Rollout/Train 概率偏差与 Entropy 漂移

更新时间：2026-07-25

## 摘要

这项排查包含两个相连但不同的问题。第一，rollout 端记录的 token 概率与 train 端对相同 token 的重算概率不一致。DSPARK verifier 没有覆盖完整状态窗口的缺陷已经修正；修正后，SGLang generation 与同引擎 full-prefix 重算在 384 个位置、49,643,520 个 logits 上逐 bit 相同。但 SGLang 与 Megatron 之间仍有约 0.044 的 mean absolute log-prob difference 和 0.020 的完整分布 TV，跨引擎残差尚未定位完。

第二，predictive DPPO 训练的 entropy 会持续上升。理论和当前逐步指标共同说明：predictive mask 只删除“继续增大 rollout/train KL”的更新，并不约束 entropy；当前被删除的更新恰好长期是降熵更新，所以 mask 系统性提高了剩余梯度的 entropy 方向。SAO/DIS 也可能改变 entropy，但其 gate 与 advantage 符号无关，因此既可能造成持续增熵，也可能造成持续降熵；旧 SAO 的先升后降曲线不能据此归因给 DIS。当前应分别完成跨引擎逐层定位，以及在相同 checkpoint、batch 和优化器状态下比较 predictive mask、无 mask 与固定-reference 约束。

## 1. 问题起源

故事始于一次异常检查。

在分析 PPO 训练的 log-prob ratio 时，我们发现了一个不寻常的模式：rollout 端（SGLang）记录的概率与 train 端（Megatron）重算的概率之间，存在系统性的偏差。所谓"系统性"，是指这个偏差并非零均值的随机噪声——它有一个可辨识的方向和量级，而不是围绕零点对称分布。

直觉上这不应当发生。Rollout 和 train 使用的都是同一份冻结的策略参数（old policy），看到的是同一个 prefix $h_t$，计算的是同一个 token $x_t$ 的概率。两个概率的比值

$$
r_t = \frac{p_{\mathrm{train}}(x_t \mid h_t)}{p_{\mathrm{roll}}(x_t \mid h_t)}
$$

在参数尚未更新的情况下理应接近 1。如果偏离 1，PPO 的 clipping 机制就会在训练开始前错误地截断梯度——训练信号从第一步起就有偏。

这个偏差到底从哪里来？有多大？对训练有什么实际影响？带着这些问题，我们做了一系列逐步收窄的实验。本文记录了这一排查过程：每一步尝试回答什么问题，排除了什么，以及什么仍然未知。

## 2. 术语与度量

在展开实验之前，先约定本文使用的几个度量。

对每个返回的 token $x_t$，定义两个概率之间的对数差和原始差：

$$
\Delta\ell_t = \log p_{\mathrm{train}}(x_t \mid h_t) - \log p_{\mathrm{roll}}(x_t \mid h_t)
$$

$$
\Delta p_t = p_{\mathrm{train}}(x_t \mid h_t) - p_{\mathrm{roll}}(x_t \mid h_t)
$$

$\Delta\ell_t$ 的正负号直接反映 train 端对"rollout 实际选中的那个 token"是更乐观还是更悲观。$\mathbb{E}[\Delta\ell_t]$ 衡量系统性方向，$\mathbb{E}[|\Delta\ell_t|]$ 衡量绝对偏离幅度。

这两个指标只反映返回 token 这一个点上的概率差异。但两个分布的差异可能不止于此——即使返回 token 的概率接近，整个词表分布可能一个更尖锐、另一个更平坦。因此我们还需要两个度量来刻画完整分布：

- **$\Delta H = H(p_{\mathrm{roll}}) - H(p_{\mathrm{train}})$**：熵差，正值表示 rollout 分布更平坦（更不确定），负值表示 train 分布更平坦。
- **TV（Total Variation）**：两个分布之间的总变分距离，衡量它们在所有 token 上的整体差异，不依赖返回 token 这一个点。

返回-token 概率的方向，和整个分布是变尖还是变平，是两个不同的问题。一个不能替代另一个。

## 3. 第一批证据：历史 H0 数据

### 3.1 数据背景

正式训练的 routing replay 机制是这样的：rollout 时保存每个 token 在每个 learned-MoE 层选中的 top-6 expert ID；Megatron 训练 forward 直接使用这些 ID，不再重新做 top-k 选择。换句话说，两边在"走哪些专家"这件事上是强制一致的。这是既定的训练配置，本文的所有分析都以此为基线。

在这批 578,893 个 response token 上，我们拿到了 rollout 时保存的 log-prob，以及 Megatron 使用相同 expert ID 重算的 log-prob。比较发现：

$$
\mathbb{E}[\Delta\ell_t] = -0.037949 \text{ nat}, \qquad
\mathbb{E}[|\Delta\ell_t|] = 0.115774 \text{ nat}
$$

负的均值意味着 Megatron 对返回 token 普遍给出了更低的概率——这是一个系统性的方向，不是均值为零的散射。

### 3.2 这个偏差是否真的影响训练？

我们做了一个最直接的因果验证：固定 128 个 sample、初始权重、优化器状态和 PPO 配置，只替换 denominator 的来源。结果如下：

| Denominator 来源 | PPO clipping 比例 | Entropy 变化 |
|---|---|---|
| Stored rollout log-prob | 8.37% | +0.154 |
| Megatron old-policy 重算 | 0.019% | -0.075 |

两列差异是明确的：那批历史 rollout log-prob 足以在参数更新前触发大量 clipping，并且改变了 entropy 的变动方向。**偏差对训练有实质影响，这一点可以定论。**

### 3.3 H0 数据的局限

但 H0 有一个关键问题：生成这批 rollout 的 SGLang 引擎当时存在自身一致性问题——它在生成过程中返回的概率，与同一个模型事后在完整 prefix 上重算的概率不一致（详见下一节）。这意味着 H0 的 stored log-prob 本身就有内生误差。

此外，H0 只保存了 log-prob ratio（一个标量），没有保存两边的原始概率或完整 logits。因此它不能回答：修正 SGLang 之后，$\Delta p$、完整分布的 entropy 或 TV 会是怎样的方向和量级。

**H0 证明了偏差存在且影响训练，但它不代表修正后的真实状况。** 要准确测量修正后的残差，必须用修复后的 SGLang 重新采集数据。

## 4. 插曲：修复 SGLang 的自身一致性

在 H0 之后的内部排查中，我们发现了 DSPARK 验证路径上的两类问题。

### 4.1 问题一：Verifier 状态覆盖不完整

DSPARK 的 target verifier 在重算概率时，没有重写全部所需的状态。直观上，verifier 应该完整复现 target model 在当前位置的 forward，但代码中有一部分状态沿用了 draft 路径的残留值——算出来的 logits 自然和实际生成时不同。

### 4.2 问题二：有限精度 Kernel 路径随工作形状变化

同一位置的概率计算，因为一次处理的 token 数量不同（生成时逐 token vs 事后重算时批量），SGLang 选中了不同的 kernel 实现。不同 kernel 在数值精度上不完全等价，最终得到不同的 logits。这不是 bug，而是有限精度下路径选择的副作用。

### 4.3 修复与验收

修正后，我们用同一个 SGLang engine 对相同位置做了两种计算：一次在生成过程中实时计算，一次在完整 prefix 上事后重算。验收标准是逐 bit 一致。

结果：在 384 个位置、完整词表共 49,643,520 个 logits 上，两种计算方式**逐 bit 完全相同**。由此计算的 log-prob difference 和 TV 均为零；公共 API 返回的 log-prob 仅有一个 $−1.19\times 10^{-7}$ 的单点差异（浮点舍入级别）。

作为对照，修正前在 block-size 4 的 sampled probe 上，3,072 个 token 的 mean absolute log-prob difference 为 0.105582，signed mean 为 −0.038707；另一组完整词表实验的 mean TV 为 0.035699。修复后这些数字全部归零。

### 4.4 这意味着什么，不意味着什么

SGLang 自身一致性的修复解决了一个内生的混淆变量：我们不再需要担心"rollout 记录的概率和 SGLang 自己事后重算的不一样"。这为后续实验提供了干净的基线。

但它**不能**证明 SGLang 与 Megatron 的概率一致。SGLang 自检通过，只是说"同一个引擎用两种方式算的结果相同"；至于 SGLang 和 Megatron 之间是否一致，需要另外的对照实验。

### 4.5 修复前后对比

DSPARK 修复前后的 bias 全景如下：

| 比较对象 | 修复前 | 修复后 |
|---|---|---|
| SGLang 自身一致性（生成 vs 事后重算） | 不一致：\|Δℓ\| = 0.106，TV = 0.036 | 逐 bit 一致（384 位置，49M logits） |
| SGLang vs Megatron（replay 开启） | \|Δℓ\| = 0.116，mean Δℓ = −0.038 | \|Δℓ\| = 0.044，mean Δℓ = +0.005 |

修复前，第二行的 0.116 实际上是两层偏差的叠加——SGLang 自身的内部不一致（第一行，~0.106）加上 SGLang 与 Megatron 之间真正的跨引擎差异。两者混在一起，无法拆开。

修复后，第一行归零，SGLang 自身的混淆变量被移除。第二行的 0.044 是干净的跨引擎残差——这就是接下来需要定位的目标。

> **注意**：修复前（H0）和修复后（H200）的 SGLang vs Megatron 数据来自不同批次的样本、不同的 LoRA 和并行配置，不是同一实验的 paired 前后对照。表中的跨行比较只反映结构和量级，不能作为 DSPARK 修复效果的精确估计。

## 5. H200 诊断：修复 SGLang 后，残差还剩多少

有了自检通过的 SGLang，我们重新采集了一批 H200 数据来测量修正后的真实残差。使用与正式训练一致的配置（routing replay 开启），同一个 base checkpoint、同一批 prefix，Megatron 回放 rollout 的 expert ID，但仍用自己的 router scores 计算 gate weights。结果如下：

| 度量 | 值 |
|---|---:|
| 返回 token：mean $\Delta\ell$ | **+0.004859** |
| 返回 token：mean $\|\Delta\ell\|$ | **0.043693** |
| 返回 token：mean $\Delta p$ | −0.000467 |
| 返回 token：mean $\|\Delta p\|$ | 0.010473 |
| 完整分布：mean $\Delta H$ | −0.002505 |
| 完整分布：mean TV | **0.019976** |
| 完整分布：top-1 相同率 | 97.40% |

几个要点：

**（1）残差确实存在。** 即使强制两边走相同的 expert ID，返回-token 的 mean absolute log-prob difference 仍有 0.044，完整分布的 TV 仍有 0.02。同 expert-ID 条件没有消除所有差异。

**（2）方向性偏倚尚不确定。** 这批数据的 mean $\Delta\ell$ 是 +0.004859，一个很小的正值。而 H0 的 mean $\Delta\ell$ 是 −0.037949，负值。两者方向相反。但这不能简单解读为"修正后方向反转了"——H0 和 H200 使用的样本、LoRA、并行配置和 SGLang 实现状态都不同，不是同一实验的前后两臂。真正的方向需要等 production-like 配置上重新采集后才能确定。

**（3）H0 和 H200 的数值不能直接比较。** H0 的 $|\Delta\ell|$ 是 0.116，H200 是 0.044，表面上下降了 62%，但两者的样本、LoRA、并行配置和 rollout 实现状态都不同。这个 62% 不能作为"DSPARK 修正效果"的定量估计。要得到有效数字，必须用同一批 prefix、同一 checkpoint 和相同 Megatron 配置，对修正前后两个 rollout 端点做 paired A/B。

## 6. Routing Replay 固定了什么，没固定什么

"Routing replay"这个名字容易让人产生"所有路由信息都被复现了"的错觉。实际上它的语义比较窄：

- **已经固定**：每个 token、每个 learned-MoE 层执行哪六个专家（expert ID）。
- **没有固定**：这六个专家的混合权重（gate weight）。Megatron 用自己的 router scores 重新计算权重——实现上等价于对自己的 scores 做 `gather(top_indices)`。
- **也没有固定**：进入 router 的 hidden state、专家内部的计算、以及其他层（attention、shared/dense、collective）的输出。

可以这样理解：replay 保证了"走同一扇门"，但不保证"每扇门走多大步"、"进门时站在哪里"，也不保证"进门之后发生什么"。

## 7. 残差从哪来：三条存活假说

当前证据不足以给剩余原因排序。以下是仍然存活的三个方向，以及两个已排除的尝试。

### 7.1 存活假说

**假说一：Gate weight 不同。** 虽然 expert ID 相同，但两边的 router scores 不同，导致六个专家的混合权重不同。目前的 artifact 没有保存 rollout 端的 gate weights，因此这一项的贡献尚未量化。这应该是下一步最先排除或确认的变量——只需在 Megatron 中做一组"IDs-only replay vs IDs+gate-weights replay"的 A/B。

**假说二：Expert 输出不同。** Rollout 使用 FlashInfer 的 MXFP4 W4A16 路径做 expert GEMM，Megatron 使用另一套解包、GEMM 和归约实现。相同输入、相同 expert ID，可能因为数值路径不同而产生不同输出。即使差异很小，经过 43 层 MoE 累积后也可能不可忽略。

**假说三：MoE 之外的状态已经不同。** Attention、shared/dense 计算、以及 collective 操作（all-gather/reduce-scatter 等）在进入或离开 learned-MoE 之前就已经积累了差异。

### 7.2 已排除的尝试

两个控制实验没有改善 mismatch，被排除在解法之外：

- **Trainer 逐 expert 的 BF16 `F.linear` 改为 grouped BF16 GEMM**：TV 没有改善。说明 expert GEMM 的调用方式（逐个 vs 分组）不是主要贡献项。
- **强行启用原本在 W4A16 配置中不执行的 shared-expert/attention FP8 路径**：mismatch 反而变大。这不是解法，而是新增了一条数值路径差异。旧 FP8 flags 现已显式关闭，仅修正配置描述，不影响正在运行的数值路径。

## 8. Predictive mask 为什么在当前训练中提高 entropy

Predictive mask 并不必然增熵。它在当前训练中增熵，是因为它持续选中了原本会让分布变尖的梯度。

canonical 在线 run `formal_r21_fp4_pp1cp2_12k_dppo_predictive_native_iter179`（W&B
`th36noa0`）的 rollout H 从 step 180 的 0.5434 升到 step 275 的 0.7137；下文的一阶
方向、固定 prompt 和冻结 replay 负责解释这条曲线，而不是只复述相关性。

在一个 response 位置上，记 rollout 分布为 $b$，train 当前分布为 $p$，采样 token 为 $k$，advantage 为 $A$，原始 importance ratio 为 $r=p_k/b_k>0$。当前实现还把它截到上限 $c$，因此实际正权重为 $w=\min(r,c)$。未 mask 的 logit 更新方向是

$$
g_k=A w(e_k-p).
$$

Predictive DPPO 估计 $D_{\mathrm{KL}}(b\Vert p)$ 沿这个更新方向的一阶变化。去掉不影响符号的正数 $w$ 后，代码中的 `predictive_dot` 是

$$
d_k=(p_k-b_k)+\sum_i p_i(b_i-p_i).
$$

当 Top-K+tail KL 已超过阈值 $\delta$，并且 $A d_k>0$ 时，该位置的梯度被删除。由于 $w>0$，截断 importance ratio 不改变这个方向判据。这个条件只回答“更新是否继续增大 rollout/train KL”，没有回答“更新是否让 $p$ 变平”。同一更新造成的 entropy 一阶变化是

$$
\left\langle \nabla_z H(p),g_k\right\rangle
=A w h_k,
\qquad
h_k=\sum_i p_i^2(\log p_i+H)-p_k(\log p_k+H).
$$

设 $M_k=1$ 表示该位置被 mask。在逐位置 logit 空间的一阶近似中，应用 mask 前后的 entropy 方向之差是

$$
\Delta H_{\mathrm{kept}}-\Delta H_{\mathrm{all}}
=-\eta\sum_{k:M_k=1} A_k w_k h_k.
$$

因此，只要被删除位置满足

$$
\mathbb{E}[A w h_k\mid M=1]<0,
$$

mask 就会系统性提高剩余更新的 entropy 方向。一个直观的二分类例子是：rollout 为 $(0.5,0.5)$，train 已经偏向第一个 token。继续强化第一个 token，或继续压低第二个 token，都会同时增大 KL、降低 entropy；predictive mask 删除这两类更新，却保留把两个概率拉近的增熵更新。它不是给 loss 添加了 entropy bonus，而是通过选择性删除梯度产生了同样方向的效果。

当前训练直接满足上述判据。step 180--272 的 93 个 step 中，`entropy/first_order_mask_delta` 93 次均为正；其均值显示，不做 mask 时的一阶 entropy 方向接近零且略为负，做完 mask 后变为正。step 274 也给出了同一方向：不做 mask 为 $-1.11\times10^{-5}$，做完 mask 为 $+3.09\times10^{-6}$，mask 单独造成 $+1.42\times10^{-5}$ 的方向偏移。这个量是同一批 logits 上的精确一阶方向，不等于经过共享参数和优化器后的实际 entropy 改变量，但它直接测量了 mask 的选择偏差。

固定 prompt 的 H200 对照进一步表明，线上曲线不只是最近 batch 更难。相同的 16 个 prompt、采样 seed 和 rollout 路径下，iter 269 相比 iter 219 的 tail mass 从 0.00705 升至 0.01076，support logit std 从 2.909 降至 2.742，sampled entropy 从 0.811 升至 0.884。两个策略生成数步后 prefix 会分叉，因此这不是 same-prefix 全词表测量；它能回答的是，后期 checkpoint 在相同 prompt 集合上确实产生了更平的生成分布。

阈值越严，更多位置更早进入上述选择过程。历史 fresh0 实验保持初始 checkpoint、PP1×CP2×EP8、global batch 256、LR、LoRA 和 Muon 配置不变，只改变 predictive KL 阈值，到 step 3 的结果为：

| 阈值 $\delta$ | step 0 entropy | step 3 entropy | 累计增幅 | step 0 超阈值 token | step 0 实际 mask |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.535 | 1.214 | **+0.679** | 19.37% | 9.99% |
| 0.10 | 0.539 | 1.003 | **+0.464** | 7.05% | 3.88% |
| 0.15 | 0.540 | 0.845 | **+0.304** | 3.46% | 1.97% |

这组实验给出了清楚的顺序：阈值越严，实际 mask 越多，entropy 上升越快。冻结 batch 的单变量 A/B 又排除了在线 batch 差异：$\delta=0.20$ 时两次更新的 entropy 增幅为 0.07597，无 mask 时为 0.01615，相差 **4.70 倍**。无 mask 时 entropy 仍会上升，所以 predictive mask 是主要放大器，不是唯一来源。证据分别见 [fresh0 threshold sweep](../../local_artifacts/deepseek-v4/r2_logs/dppo_threshold_fresh0_sweep_20260721.txt) 和 [predictive mask 对无 mask 的冻结-batch A/B](../../local_artifacts/deepseek-v4/r2_logs/entropy_ab_wave1_mask_vs_nomask_20260721.txt)。

这种局部 KL gate 也不能阻止累计漂移：每轮 rollout 都来自不断更新后的策略，锚点随策略一起移动。即使每一步 train 都接近当轮 rollout，数百步后仍可远离初始策略。要限制长期 entropy，必须增加固定-reference KL、明确的 entropy 约束，或用带恢复力的平滑约束替代 hard deletion。

## 9. SAO/DIS 会不会让 entropy 持续上升或下降

会，两种方向都可能，但 DIS 没有 predictive mask 那样固定的方向选择。这里分析的是当前代码中的 DIS policy loss，而不是 SAO 的 value model 和 single-rollout 等其他组件。

DIS 根据 sampled-token ratio

$$
r_t = \frac{p_{\mathrm{train}}(x_t\mid h_t)}{p_{\mathrm{roll}}(x_t\mid h_t)}
$$

是否落在区间 $(1-\epsilon_l,1+\epsilon_h)$ 内决定去留。其 mask 为

$$
M_t^{\mathrm{DIS}}
=\mathbf{1}\!\left[r_t\le 1-\epsilon_l\ \text{or}\ r_t\ge1+\epsilon_h\right].
$$

DIS 删除梯度造成的 entropy 方向变化仍为

$$
\Delta H_{\mathrm{DIS}}-\Delta H_{\mathrm{all}}
=-\eta\sum_{t:M_t^{\mathrm{DIS}}=1} A_t r_t h_t.
$$

所以，被删除更新平均降熵时，DIS 使剩余梯度更增熵；被删除更新平均增熵时，DIS 使剩余梯度更降熵。两种情况在理论上都成立。关键区别是，predictive mask 的判据包含 $A_t$ 的符号，只删除继续增大 KL 的方向；DIS 的 gate 与 advantage 符号无关。在同一个高 ratio 位置，$A>0$ 的外推梯度和 $A<0$ 的纠正梯度都会被删除。DIS 因而没有内建的 entropy 方向，但会按 ratio 截断 action 和 advantage 的联合分布，使原本可能抵消的梯度失去平衡。

DIS 也没有把策略拉回 rollout 的恢复力。一个 token 一旦越界，外推和纠正梯度都为零；只有后续 rollout 更新、其他 token 的共享参数更新或样本组成变化，才可能让它重新进入有效区间。因此，如果有效梯度长期偏向同一个 entropy 方向，DIS 完全可能允许 entropy 持续上升或持续下降。

旧 SAO run 只能说明发生过三个阶段，不能确定 DIS 是方向来源。step 0--16，entropy 在 0.41--0.60，DIS 有效 token 比例为 98.0%--99.3%。step 16--24，entropy 从 0.410 升至 0.980，但峰值仍有 98.0% token 有效；大量 DIS masking 不是这次上升的必要条件。step 24--33，entropy 从 0.980 降至 0.038，同时 rollout 平均 log-prob 从 -1.04 升至 -0.11、平均输出长度从 6.5k 增至 9.5k、截断率从 8.6% 增至 77.3%，说明策略整体进入了高置信、超长输出状态。

旧 run 的 Muon spectral update scale 为 1.0，是当前配方 0.18 的 5.56 倍，同时还有其他优化器配置差异。它是 boom-bust 的主要候选原因，但没有完成只改变 Muon scale 或只开关 DIS 的单变量 A/B。因此现有证据不能判断 DIS 对上升和下降各贡献多少，也不能声称 SAO 必然产生某个 entropy 方向。原始逐步指标见 [SAO/DIS boom-bust evidence](../../local_artifacts/deepseek-v4/r2_logs/dis_muon_boom_bust_and_restart_20260722.txt)。

## 10. 下一步

建议按以下顺序收窄：

**第零步（前置条件）：** 采一批 production-like 的完整数据。使用已通过自检的 SGLang，同时保存每个位置的 returned-token log-prob、完整 logits、expert ID 和 gate weights。在这批数据上重新确定 $\Delta\ell$、$\Delta p$、$\Delta H$ 和 TV 的方向与幅度。这是后续所有结论的基线。

**第一步：量化 gate weight 贡献。** 在 Megatron 中比较两个单变量配置——只 replay expert ID vs 同时 replay expert ID 和 rollout gate weights。两者的差值直接度量 gate weight mismatch 的贡献。

**第二步：逐层定位分叉点。** 在相同 expert ID（必要时相同 gate weight）的条件下，从 layer 0 开始逐层比对 hidden state。在首次分歧处，检查该层各组件的输入/输出以及 collective 边界。

**第三步：分解 expert 内部差异。** 对 learned-MoE 层，分别保存 expert 输入、单个 expert 输出、以及加权合并前后的结果，将 W4A16 expert arithmetic 的差异与归约（reduce）的差异分开。

**第四步：量化 entropy 约束的因果贡献。** 在当前 checkpoint、同一冻结 batch 和优化器状态上比较 $\delta=0.15$、无 predictive mask、平滑 KL penalty 和固定-reference KL。若继续追 SAO 的 boom-bust，则分别做“只改变 Muon scale”和“只开关 DIS”的 A/B，并按 upper/lower ratio、advantage 正负和 $A r h$ 符号报告被删除的 entropy contribution。

端到端验收应使用修正后的 production-like 配置，同时报告：返回-token signed/absolute $\Delta\ell$、原始 $\Delta p$、完整词表 TV/entropy，以及参数更新前的 PPO clipping 比例。

## 附录：证据与实现入口

### 当前正式训练

predictive DPPO 分支最后保存的是 `iter_0000279`。step 279 的 train entropy 为 0.708，rollout entropy 为 0.694，train/rollout mean absolute log-prob difference 为 0.0635，Top-K KL 为 0.00937。该进程随后生成过 rollout 280，但没有训练它，也没有保存新的 checkpoint；切换目标函数时丢弃了这批未提交数据。

2026-07-25 10:26 JST，训练从 `iter_0000279` 的 LoRA、optimizer、RNG 和 dataset state 恢复，并把 policy loss 切换为 SAO 的 token-level DIS。当前参数是 `eps_clip=0.80`、`eps_clip_high=3.0`；两者都是相对 1 的偏移量，因此 DIS 保留的 ratio 区间是

$$
1-0.80 < r < 1+3.0,
\qquad\text{即}\qquad
0.2 < r < 4.0.
$$

首个新批次 rollout 280 的 rollout entropy 为 0.691。对应 train step 280 已成功完成：train entropy 为 0.705，train/rollout mean absolute log-prob difference 为 0.0627，PPO KL 为 0.00988；DIS 保留了 99.994% 的 token，upper/lower clip 比例分别为 $3.13\times10^{-5}$ 和 $2.06\times10^{-5}$。日志已出现 DIS 指标，且不再出现 predictive-mask 指标，证明目标函数切换生效。单个 step 只能验收恢复与配置，不能判断 SAO 后续会持续增熵还是降熵。截至 2026-07-25，该阶段 W&B run 为 [qsnifb60](https://wandb.ai/shuailin_chen/slime/runs/qsnifb60)。

- [H200 train/rollout full-vocabulary probe](../../local_artifacts/deepseek-v4/r2_logs/train_rollout_full_vocab_h200_20260723.txt)：route-replay 配置下的完整指标、artifact 和 SHA256。
- [最新 predictive entropy 证据](../../local_artifacts/deepseek-v4/r2_logs/predictive_entropy_rise_iter219_269_20260725.txt)：step 180--275 的 mask 方向、固定 prompt iter 219/269 对照、artifact 和 SHA256。
- [SAO/DIS 从 iter279 恢复的验收证据](../../local_artifacts/deepseek-v4/r2_logs/sao_resume_iter279_20260725.txt)：实际恢复组件、首个 rollout 280 和 train step 280 的完整关键指标。
- [PPO denominator-only fixed replay](../../local_artifacts/deepseek-v4/r2_logs/entropy_ab_wave4_denominator_only_20260721.txt)：历史 H0 mismatch 对 clipping 与 entropy 的因果影响。
- [SGLang generation 与同模型事后重算的对照证据](../../local_artifacts/deepseek-v4/r2_logs/tim_h200_dspark_root_fix_20260722.txt)：DSPARK 算法流程错误、有限精度路径和修正后逐 bit 验收。
- [DSPARK verifier-window patch](../../scripts/dsv4/patches/dspark_port_series/0014-port-dspark-runtime-verify-rewrite-window.patch)：把运行时 `verify_width` 传到 compress planner，并据此重写完整窗口。
- [Routing replay 实现](../../slime/utils/routing_replay.py)：`replay_forward` 回放 expert ID，并从 Megatron scores gather gate weights。
- [Fixed replay launcher](../../scripts/dsv4/studies/entropy/run_entropy_fixed_batch_arm.sh)：显式设置 `USE_ROLLOUT_ROUTING_REPLAY=1` 并审计 route payload。
- [H200 probe 脚本](../../scripts/dsv4/diagnostics/mismatch/train_rollout_full_vocab_probe.py)。
- [VeXact](https://arxiv.org/abs/2605.14220)：同权重、同输入下 rollout/trainer mismatch 与 paired-logit 验收。
- [Tree-Based Invariant Kernels](https://arxiv.org/abs/2511.17826)：通过固定归约结构减少并行形状变化引入的数值差异。
