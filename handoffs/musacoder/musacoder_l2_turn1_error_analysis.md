# Level2 Turn-1 错题分析

## 结论

L2（新 shape、100 题 × 8 samples、load_inline、train-mode、TF32-off、1e-4）turn-1 共 800 样本，correct 513（64.12%），**失败 287 个**。失败不是均匀分布的"能力噪声"，而是由**五个可命名的机制**构成，其中前两个（BatchNorm 语义陷阱 + 大 shape 下的 300s task 超时）合计约 105 个样本（~37% 的失败）是**口径/预算类**问题，不是模型写错 kernel；真正的模型硬错集中在少数题上。

| 失败类别 | 样本数 | 占 800 | 机制性质 |
|---|---:|---:|---|
| output_mismatch（真数值错，max_diff>1e-2） | 151 | 18.9% | **其中 76 个（一半）来自 11 道 BatchNorm 题**的 train/eval 语义陷阱；其余 75 个是真实现错（LayerNorm 陷阱等） |
| near_miss（max_diff≤1e-2） | 53 | 6.6% | 容差敏感：大规模 fp32 归约换序/精度漂移，放宽到 1e-2 全部转正 |
| task_timeout（300s task 超时，执行阶段主导） | 29 | 3.6% | shape 驱动的预算问题：同一代码、同一预算，换旧（小）shape 后 31/32 编译且正确（见机制二的更正说明） |
| runtime_exception（forward 阶段异常） | 22 | 2.8% | wrapper/launch 逻辑错（TypeError、CUDA error 等） |
| compile_fail（真编译错） | 19 | 2.4% | 真语法/binding 错，集中在 2 题（pid56 6 个、pid40 5 个） |
| shape_mismatch（输出形状错） | 13 | 1.6% | 输出尺寸公式算错，集中在 pid75（5 个） |

## 机制一：BatchNorm train/eval 语义陷阱（最大单一来源，~76 样本）

11 道含 BatchNorm 的题（pid 11/15/33/39/41/52/72/73/77/84/97）turn-1 只对 5/88（5.7%）。机制已实证（详见 `musacoder_load_inline_eval.md` L2 差距分析一节）：

- 模型系统性地按**推理期语义**实现 BN——读 `self.bn.running_mean/running_var`（pid73 的 kernel 注释原文就写着 "using running stats for inference"），这是真实部署代码的标准写法；
- 而 KernelBench 评测约定（官方与我们一致）从不对 reference 调 `.eval()`，reference 用**当前 batch 统计量**。两者是本质不同的计算，必然 mismatch。
- 对照证明非重打分伪影：reference 换 `.eval()` 后这组 5.7%→76.1%；GroupNorm/InstanceNorm 题（无 running stats）70.6% 完全不变。

**定性**：不是"不会写 BN kernel"，是"按更常见的语义写了 BN"。这是 KernelBench train-mode 约定与模型先验的冲突。

## 机制二：300s task 超时 = 大 shape 下执行阶段烧穿预算（29 样本）

集中度极高：pid100 8/8、pid78 4/8、pid32 3/8、pid39/46/98 各 2/8。已实证是 shape 驱动（非模型错）：把**同一份 response** 换成 2025-07 放大前的旧 reference、**同样 300s 预算**重打分，pid40/56/78/100 合计 32 样本中 31 个从"超时"变"编译且正确"。

**机制注意（已更正）**：这不是"nvcc 编译超过 300s"——同一份源码两种 shape 下编译时间相同（输入维度是运行期参数）。300s 是 **task 级预算**（error_message="Task ... timeout after 300s"，compiled=False 只表示 task 没跑完）：编译固定吃掉 ~90s（与 shape 无关，见 `musacoder_l1_l2_turn1_time_breakdown.md`），剩余预算被 correctness forward 烧穿——朴素融合 kernel 在大 shape 输入上单次 forward 可达分钟级，小 shape 下毫秒级。600s 隔离重跑会给出实测阶段拆分。

## 机制三：容差敏感的 near-miss（53 样本）

max_diff 在 1e-4 与 1e-2 之间。典型：pid66（8/8）、pid13/14/19（各 5）。代表性根因（pid14，已读代码证实）：模型做了**数学上等价**的化简（先把 weight 按 hidden 维求和再点积，等价于 matmul 后求和），8192×8192 归约换序改变 fp32 累加误差——换旧（小）shape 后 8/8 全部转正。这类属于"数学正确、fp32 顺序敏感"，官方旧口径（1e-2）下全部计为正确。

注：pid66/83 含 `nn.Dropout(p=0.2)`，train-mode reference 的 dropout 是随机的，确定性 custom kernel 结构上不可能逐元素 match；这 2 题构成 ~2pt 的口径天花板损失。

## 机制四：真实模型错误（少数题上的硬错，~75 样本非 BN 的真数值错 + 19 真编译错 + 22 runtime + 13 shape）

- **pid3、pid34（LayerNorm 陷阱，各 8/8 全错）**：`ConvTranspose3d` 后接 `nn.LayerNorm(out_channels)`，而输出宽度 W′ 恰好数值上等于 out_channels——reference 实际按**宽度维**归一化。模型推理原文写 "LayerNorm is applied over the channel dimension"，按通道归一化，落入参数巧合构成的陷阱。跨新旧 shape、跨两次独立生成都稳定复现——真实推理错误。
- **真编译错集中在 pid56（6/8）、pid40（5/8）**：语法/binding 硬错。
- **runtime_exception 集中在 pid38（4）**、pid6/94（各 3）：wrapper 参数/launch 配置错。
- **shape_mismatch 集中在 pid75（5/8）**：输出尺寸公式算错。

## 12 道 0/8 全错题的归因总表

| pid | 题 | 归因 |
|---|---|---|
| 11/15/33/41/72/73/77 | BatchNorm 系列 | 机制一（BN 语义陷阱） |
| 66 | Matmul_Dropout_Softmax | 机制三注（train-mode 随机 dropout 结构性不可解） |
| 3/34 | LayerNorm 陷阱 | 机制四（真实推理错误） |
| 14 | Gemm_Divide_Sum_Scaling | 机制三（fp32 归约换序精度，旧 shape 下 8/8 正确） |
| 100 | ConvTranspose3d_Clamp_Min_Divide | 机制二（task 超时，旧 shape 下 8/8 编译且正确） |

## 读法

- 每题正确数直方图：0/8 有 12 题、8/8 有 31 题、7/8 有 14 题——失败高度集中，非弥散。
- 若按"机制一二三都算口径/预算因素"的最宽松口径修正，T1 上限约 81%（见主 handoff 的叠加估算）；剩余 ~12pt 差距是真实模型错误 + 无法验证的官方侧因素（prompt/MUSA 工具链）。
- `.so cannot open` 并发假失败 = 0（L1 时代的 bug 已根治，本次干净）。

<!-- 证据与复现：
- 分类脚本：.22:/tmp/l2_error_attr.py（按 correctness/compiled/error_message/correctness_issue/max_difference 分类，逐题聚合）。
- 数据源 dump：/nfs/FM/chenshuailin/projects/kernel_agents/slime-musacoder-mt/experiments/EvalMT3.load_inline.MusaCoder-27B.CTX40960/MusaCoder-27B.mt3_level2.load_inline/dumps/rollout_data/eval_0.pt（turn_idx==0 的 800 条）。
- 类别计数交叉核对：codex 独立分类（correct 513 / real 164 / near-miss 53 / timeout 29 / runtime 22 / compile 19）与本表（151/13 = 其把 13 个 shape_mismatch 并入 real 164）一致，仅分桶粒度不同。
- BN 机制、编译超时 shape 实证、pid3/34/14 逐个读代码的证据链见 musacoder_load_inline_eval.md 的 L2 差距分析（一/二/三/四节）及其证据注释。
- 时间/编译成本背景见 musacoder_l1_l2_turn1_time_breakdown.md。
-->
