# 多轮Rollout加速：Prefix Cache 命中率优化

> **实验环境**  
> 模型 Qwen3.6-27B (16 full-attn + 48 linear-attn/mamba, hybrid)  
> SGLang 0.5.12.post1 · CUDA 12.9  
> 评测 KernelBench level1, 800 samples, ctx65536 / n8 / summ1600 / max_turns=3 / TP2（除非注明）

---

## 1. 结论

**尝试的两条路线都能提高 prefix cache 命中率，但都没能让 rollout 明显加速，甚至还会带来精度的下降，因此本方向无功而返。**

核心原因：这个 workload 的 wall time **不由 prefix cache 决定**。

| wall time 组成               | 受 cache 影响？ | 典型耗时                    | wall 占比 |
|------------------------------|:--------------:|-----------------------------|:---------:|
| **prefill**                  | ✅             | prompt 几k tok @ 5k-20k tok/s ≈ 亚秒~2s | **小** |
| **decode**                   | ❌             | response 4-8k tok @ ~1400-1850 tok/s ≈ 数秒 | **大** |
| **KernelGym 编译 + benchmark** | ❌           | 编译 + 30 warmup + 50 perf trials ≈ 数秒~数十秒/kernel | **大** |

命中率从 21.5% 翻倍到 53%，但 s/it 和 wall 基本不变：

| run                  | chat-tempalte    | preserve | hicache | prefix hit | s/it | wall    | resp_len | score     |
|----------------------|---------|:--------:|:-------:|----------:|-----:|--------:|---------:|----------:|
| baseline（TP2, 4 eng）| 官方    | ❌       | ❌      | 36.01%    | 8.23 | 1:49:40 | 8453     | **0.379** |
| TP4（2 engine）       | 官方    | ❌       | ❌      | 44.81%    | 8.90 | 1:58:37 | 8526     | 0.375     |
| stock-preserve       | 官方    | ✅       | ❌      | 21.5%     | 8.45 | 1:52:39 | 4466     | 0.34      |
| cachetmpl            | no-norm | ✅       | ❌      | ~44.5%    | ~8   | —       | —        | -      |
| cachetmpl + hicache  | no-norm | ✅       | ✅      | ~53%      | ~8   | —       | —        | -      |

<!-- ### 如何判优

- ❌ **别看 prefix cache hit**——它量的是 prefill 复用比例，不代表速度。
- ❌ **wall / s-it 区分度也低**——都 ~2h，被 decode + KernelGym 主导。
- ✅ **看 score（质量）。** -->

### 两个反直觉现象

1. **preserve 让 response 变短**（8453 → 4466），模型看到上轮 reasoning 后不重复推导 → decode 减半，正好抵消更差的 prefill → 命中腰斩反而略快。
2. **hicache 的高命中是用 host ↔ device 搬运换的**，搬运开销吃掉了 prefill 节省。

<!-- ### 如果要加速这个 workload

**应从 decode（生成更短/更快）或 KernelGym（评测更快）下手，而非 cache。**

下面两条路线的价值仅在于"当你因 reward/质量原因必须开 preserve 时，让缓存别崩"——但不会让 wall 变快。 -->


---

## 3. 路线 1：TP2 vs TP4（增加 engine 数 vs 提高命中）

8 卡固定：TP2 = 4 个 engine，TP4 = 2 个 engine。

<!-- ### 3.1 机理 -->

<!-- `sglang_rollout.py:352-355`：每个 sample 用独立 uuid 作 session_id，`--router-policy consistent_hashing` 按 uuid 路由。同一 prompt 的 n8 样本独立散到各 engine。 -->
engine 越少，共享前缀冗余 prefill 越少，单 engine radix 覆盖工作集更大 → 命中更高。且更少的engine数意味着负载均衡更容易。

> <!-- 实现细节 -->
<!-- > 多轮内同一轨迹 session_id 跨轮不变 → 粘同一 engine。 -->

### 3.2 实测

| 指标                | baseline TP2（4 eng）| TP4（2 eng）        |
|---------------------|--------------------:|-------------------:|
| prefix_cache_hit    | 36.01%              | **44.81%**         |
| wall time           | **1:49:40**         | 1:58:37            |
| per-engine decode   | 1024 tok/s          | **1781 tok/s**     |
| aggregate decode    | **4×1024 = 4096**   | 2×1781 = 3562      |

> <!-- 置信度说明 -->
<!-- > confounder（TP4 并发 96 > 48、mem 0.85 < 0.9 都在压低 TP4 命中）反而让"TP4 命中更高"成为**减少 engine 提命中的强证据**。 -->

### 3.3 结论

**减少 engine 确实提命中（+8.8pp），但 TP4 更慢、不值。** 见§1 表格：TP4 命中 44.81% > baseline 36.01%，但 wall 1:58:37 > 1:49:40，score 基本持平。

TP 2→4 每 engine 只提速 1.74×（亚线性，不到 2×），补不回 engine 数减半 → TP2 聚合 decode 快 ~15%。

> <!-- 亚线性原因 -->
<!-- ~26% 损耗来自：
- custom_all_reduce 在 TP4 被关
- 48 个 mamba 层细切 kernel 效率低。 -->

<!-- **原则：用能装下模型 + KV/mamba 的最小 TP，剩余 GPU 全开数据并行 engine。** 27B 在 TP2 能装下 → TP4 除了"每 engine 更多显存 headroom"外只带来通信开销。

**更优的提命中杠杆（未实现）**：prompt 级路由——把 session_id 从 per-sample uuid 改成 per-prompt id，让同一 prompt 的 n8 集中到一个 engine，不动 TP/不牺牲吞吐就把共享前缀冗余 prefill 压到最低。 -->

> <!-- 注意事项 -->
<!-- > 代价是单 prompt 负载集中，需关注 router `balance_abs_threshold=10` / `balance_rel_threshold=1.2`。 -->

---

## 4. 路线 2：preserve_thinking + hicache

> **前置结论：preserve_thinking 不仅没加速，还降低了精度。** 见§1 表格：stock-preserve 的 score 仅 0.34，显著低于 baseline 的 0.379。原因分析见 §4.3。

### 4.1 背景：为什么默认命中低

**默认行为（baseline）**：历史轮 assistant 不带 thinking。模型上一轮生成的 `<think>...</think>` reasoning 不会进入下一轮 prompt。radix 无从复用上轮生成。

<!-- **根因**：官方 chat template（L100-103）只有最新轮或 `preserve_thinking=true` 才保留 `<think>`；而 drkernel 每轮 assistant 后跟一条 feedback 使 `last_query_index` 越过最新 assistant → 历史轮 thinking 被删。 -->

<!-- **实证**（`scripts/analysis/preserve_thinking_prefix_match.py`）：
```
preserve_thinking=False: radixLCP=28/185,  generated_reused=0/155,   diverge@ '<think>' vs 'Here'
preserve_thinking=True:  radixLCP=185/185, generated_reused=155/155
encode(decode(G))==G ? True   ← 不是 BPE 重分词的锅
``` -->

> <!-- SSM 缓存补充 -->
> device radix 本就缓存 mamba state（`mem_cache/mamba_radix_cache.py`）；SSM 缓存不缺，被同一 prefix-match 卡住。

### 4.2 修复方案

1. 启用 preserve_thinking
2. 官方chat-template会重新归一化 thinking（split/rstrip/lstrip/trim/强制 `\n</think>\n\n`），这部分逻辑需要删除
<!-- #### (a) 快速启用 preserve_thinking

```bash
--apply-chat-template-kwargs '{"preserve_thinking": true}'
``` -->

<!-- 通过 `arguments.py:543` → `materialize_prompt:196` 透传。 -->

<!-- **问题**：官方模板的"保留"分支会重新归一化 thinking（split/rstrip/lstrip/trim/强制 `\n</think>\n\n`），模型真实输出常对不上 → 仍会断前缀。 -->
<!-- 这就是 **stock-preserve 命中只 21.5%** 的原因。 -->

<!-- #### (b) no-norm 模板（根治归一化） -->

<!-- `slime_plugins/drkernel/prompt_templates/chat_template_no_think_norm.jinja` -->

<!-- 最小 diff：仅 assistant keep 分支改成**逐字渲染**（不 split/trim/强制空白），其余（gating、tool_calls、multi_step_tool 扫描）与官方一致。仍需 `preserve_thinking=true`。 -->

<!-- **单测**：`tests/utils/test_drkernel_cache_template.py`（6 边界 case + 对照 100% 命中，15 passed）。 -->

<!-- **部署**：用 symlink model dir 技巧——原模型目录全 symlink，只把 `chat_template.jinja` 换成 no-norm 模板，launcher 改 `MODEL_DIR` 指过去即可。 -->

> <!-- 技术细节 -->
<!-- > transformers 加载时 `.jinja` 文件优先于 tokenizer_config.json 内联字段（已实测）。 -->

> <!-- 已清理 -->
<!-- > dead flag `--preserve-history-thinking`（drkernel）从没接线，已删（args.py + design-docs）。 -->

> <!-- 备选方案（未实现） -->
<!-- > token 续接：保留 `prev input_ids + 生成 token_ids` 直接拼接、不重渲染 → LCP 222/222，免疫所有模板/分词边界问题（含截断）。但同样把 reasoning 留上下文。 -->

### 4.3 代价：preserve 撑大上下文 → 显存不足

preserve 把历史 reasoning 全留 → 常驻上下文显著变大。

**Prompt token 数（进入该轮的 context）— p50 / p90 / p99**

|        | baseline                | PRESERVE                       |
|--------|--------------------:|-------------------------------:|
| turn-1 | 1423 / 1690 / 1869  | 1423 / 1690 / 1869             |
| turn-2 | 3311 / 4776 / 5922  | **13743 / 28385 / 58699**      |
| turn-3 | 5105 / 7831 / 10155 | **18666 / 40235 / 65615**      |

**Answer token 数（该轮生成，含 thinking）— p50 / p90 / p99**

|        | baseline                  | PRESERVE                    |
|--------|-----------------------:|----------------------------:|
| turn-1 | 12224 / 34602 / 62304 | 11929 / 26386 / 56986       |
| turn-2 | 8287 / 12026 / 24617  | **4765 / 8157 / 27163**     |
| turn-3 | 8045 / 11877 / 29689  | **4096 / 6780 / 16917**     |

**findings**：

<!-- - turn-1 两者相同（sanity check）。 -->
- preserve 的 turn-2/3 prompt 是 baseline 的 ~4 倍；**turn-3 p99 已达 65615，顶满 64k ctx**。
- preserve 的 turn-2/3 生成只有 baseline 的 ~一半——模型看到上轮思考，follow-up 更短。
- 即：preserve **用"更短的生成"换"更长的 context"**；后者正是显存压力来源。
- **精度下降**：大上下文 + 截断导致后轮提升能力受损，score 从 baseline 的 0.379 降至 0.34（见§1 表格）。

> <!-- 峰值上下文数据 -->
> bootstrap 峰值上下文（= turn-3 prompt + turn-3 生成，`scripts/analysis/preserve_thinking_budget.py`）：均值 baseline ~21k / PRESERVE ~36k，p95 PRESERVE ~64k（贴边）。

**大上下文 → KV/mamba 池被撑满（usage 冲到 0.96）→ 已完成的续轮被 LRU 淘汰**。 这就是路线 2 走到"需要 hicache"的原因。

### 4.4 hicache（host KV/mamba offload）

<!-- 之前以为 hicache "启用即崩"——**已更正**：崩的根因是 **`hicache-ratio` 太大**。 -->

<!-- | hicache-ratio | host 内存需求                    | 结果     |
|:-------------:|--------------------------------|---------|
| 2.0           | KV 46.45GB + Mamba 41.64GB     | 首次 prefill 即崩 |
| 1.2           | KV ~29GB + Mamba ~26GB         | **全程正常** |
| 1.05          | KV 24.38GB + Mamba 21.86GB     | 正常     |

**建议：用 hicache 时 ratio 压到 ~1.2。** -->

**效果**：cache 命中率从 ~44.5% 提升至 ~53%（见§1 表格 cachetmpl vs cachetmpl+hicache），但 wall/s-it 仍无变化。hicache 对这个 workload 的 wall **没帮助**，只在"既要 preserve、又想省点 prefill 算力"时才值得开。
<!-- 。比早期"+4pp"高，因为早期是官方模板、续轮前缀本就断，缓存了也对不上；修好模板后 hicache 才有可取回的东西。 -->

<!-- **但 hit↑ ≠ 更快**：53% 命中的 run，wall/s-it 仍和 21.5% 的一样。hicache 对这个 workload 的 wall **没帮助**，只在"既要 preserve、又想省点 prefill 算力"时才值得开。 -->

> <!-- 旧弯路（不用再查） -->
<!-- > driver 4GB cudaHostAlloc 限制（[#7666](https://github.com/sgl-project/sglang/issues/7666)，已被 575.51.03 修复，且用的是 cudaHostRegister）、容器 `ulimit -l=64KB`（CUDA 走 uvm 直接 pin，bypass RLIMIT_MEMLOCK，无需 `--cap-add IPC_LOCK`）——都不是崩因。 -->

<!-- 
## 5. 实验路径索引

| 标签                 | 路径                                                                                                              | 关键配置                                       |
|----------------------|------------------------------------------------------------------------------------------------------------------|-----------------------------------------------|
| baseline（TP2）      | `checkpoints/Qwen3.6-27B/20260528_043455_ctx65536_n8_summ1600_nonla_emfrac09_newSlimeKG/`                        | TP2(4 eng) / mem0.9 / 并发48                    |
| TP4 对照             | `checkpoints/Qwen3.6-27B/20260528_234734_ctx65536_n8_summ1600_nonla_emfrac09_newSlimeKG_tp4/`                    | TP4(2 eng) / 官方模板 / 无 preserve / 无 hicache |
| stock-preserve       | `checkpoints/Qwen3.6-27B/20260529_022725_ctx65536_n8_summ1600_nonla_emfrac09_newSlimeKG_tp4_preserve_think/`     | 官方模板 + preserve_thinking                    |
| cachetmpl            | `checkpoints/Qwen3.6-27B-cachetmpl/20260529_045754_ctx65536_n8_summ1600_nonla_emfrac09_newSlimeKG_tp4_thinkv2/`  | no-norm 模板 + preserve                        |
| cachetmpl + hicache  | `checkpoints/Qwen3.6-27B-cachetmpl/20260529_054917_..._tp4_thinkv2_hicachev2/`                                   | no-norm + preserve + hicache ratio1.2           |
| hicache 早期崩溃     | `checkpoints/Qwen3.6-27B/20260528_081129_*_hicache/run.log`                                                      | ratio 2.0 启用即崩（见 §4.4）                   |

> <!-- symlink model dir 技巧 -->
<!-- > 自定义模型目录 `Qwen3.6-27B-cachetmpl/` = 原模型目录全 symlink，只把 `chat_template.jinja` 换成 no-norm 模板；launcher 改 `MODEL_DIR` 指过去即可，无代码改动。 -->


<!-- ## 6. 关键源码定位

| 路径 | 关注点 |
|------|--------|
| `slime_plugins/drkernel/rollout.py:188-203`       | `materialize_prompt`：重渲染 + 透传 kwargs |
| `slime_plugins/drkernel/rollout.py:732-825`       | 多轮循环：L806 append assistant、L823 append tool、L824 重渲染 |
| `slime/rollout/sglang_rollout.py:191,216,352-355` | `input_ids=encode(prompt)`（每轮重分词）；per-sample uuid session_id |
| `slime/utils/arguments.py:543`                    | `--apply-chat-template-kwargs`（json.loads） |
| `dumps/tokenizer/chat_template.jinja:94-103,147-152` | thinking 拆分/删除 + `<think>` 注入 |
| `slime_plugins/drkernel/prompt_templates/chat_template_no_think_norm.jinja` | no-norm 修复模板 |
| `/sgl-workspace/sglang/.../mem_cache/mamba_radix_cache.py`    | device 端 hybrid radix cache |
| `/sgl-workspace/sglang/.../mem_cache/hi_mamba_radix_cache.py` | mamba host pool（PR #20457） |

> <!-- 不常用 -->
<!-- > `/sgl-workspace/sglang/.../managers/schedule_batch.py:1730-1732`：hicache sticky error 浮出处（非真正失败点）。 --> 

## 7. 给下一个调查者的建议

2. **`#cached-token=0` 大多不是 miss**：① 分块尾（长 prompt 按 8192 切，匹配只记在 chunk#1）② run 启动 warmup ③ usage 饱和淘汰。请求级命中比按行算的高。
3. **`sample=` 索引会重复**，per-trajectory 聚合不可靠；用 prefill 日志 `#cached+#new` 或 per-turn resp_len 边际分布。
4. **要拆 device vs host 命中**，需先 patch slime 持久化 `meta_info["cached_tokens_details"]`（当前只存总 cached_tokens，`slime/utils/types.py:102-105`；SGLang 计算在 `schedule_batch.py:1819-1841`）。
5. hicache 不要太多 commit 到 0.5.12.post1，等 0.5.13+（[#12826](https://github.com/sgl-project/sglang/issues/12826) 在推进 hybrid hicache）。

<!-- --- -->

<!-- ## 8. 相关资源

| 类型 | 路径 |
|------|------|
| 复现脚本 | `scripts/analysis/preserve_thinking_prefix_match.py` |
| 预算分析 | `scripts/analysis/preserve_thinking_budget.py` |
| 每轮长度 | `scripts/analysis/per_turn_len.py` |
| 每轮精度 | `scripts/analysis/per_turn_acc.py` |
| 单测 | `tests/utils/test_drkernel_cache_template.py` |
| Ablation launcher | `scripts/eval_drkernel/rollout_speedup_ablation/debug.27b.hicache_ablation.sh` |
| 主 rollout handoff | `handoffs/complete/handoff_drkernel_w8a8_rollout.md` | -->
