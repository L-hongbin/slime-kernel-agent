# Rollout 加速：投机解码（SpecDec）

> **环境**  
> 模型 Qwen3.6-27B hybrid（16 full-attn + 48 linear-attn/mamba），BF16 精度  
> 8× A800-80GB · SGLang 0.5.12.post1 · CUDA  
> 评测 KernelBench L1, 800-sample eval, 多轮 kernel 编辑（max_turns=3）

<!-- 属 `handoff_rollout_speedup.md` 第 4 条方向。wall 真瓶颈 = decode 量 + KernelGym 外部评测；specdec 只加速 decode 段，wall 收益会被外部评测摊薄（如 w8a8 per-token 1.52× → wall 仅 1.07×）。 -->

---

## 1. 结论

**EAGLE 已实测 ~1.23× wall 加速且 score 无损，是目前所有方向里最干净的加速。** 还有 tree 化 / NGRAM / EAGLE3 三条未尽路线值得试。

| run | spec | wall（800-eval） | score | decode tput |
|-----|------|:----------------:|------:|------------:|
| EAGLE 链式 | `num-steps=3, topk=1, draft=4` | **1:36:09** | 0.381 | ~2033 tok/s |
| 无 spec（baseline） | — | 1:58:37 | 0.375 | ~1781 tok/s |

- **wall 1.23×**（7117s → 5769s），score 基本持平（投机解码无损，符合预期）。已优于 w8a8 的 1.07× wall。
- `spec_accept_length ≈ 3.23`（accept rate 0.68–0.99），**已逼近 `num_draft_tokens=4` 天花板 → 链式配置饱和**。再提只能加宽 / 换更强 draft / 换机制。

<!-- 注意：两 run 的 `mem-fraction`（0.85 vs 0.9）不完全受控，不是严格 A/B，但同模型/同数据/同 800-eval，方向可信。decode tput 均值因 batch 混杂噪声大，以 wall 为准。 -->

<!-- 脚本：`scripts/eval_drkernel/rollout_speedup_ablation/debug.27b.tp4.eagle.sh`（链式：EAGLE, num-steps=3, eagle-topk=1, num-draft-tokens=4）。 -->

<!-- Run 路径：
- EAGLE: `20260529_075505_*.tp4.eagle`
- baseline: `20260528_234734_*_nonla_emfrac09_tp4` -->

---

## 2. 未尽路线（按优先级）

| 优先级 | 路线 | 训练？ | 预期收益 | 关键约束 |
|:------:|------|:------:|---------|---------|
| **P0** | EAGLE **tree 化** | 无 | accept 3.2→4~5 | 低并发收益大；高并发（batch 96）时验证算力是否反噬 |
| **P0** | **NGRAM / suffix** | 无 | 复刻段 accept 10~18 | 需关 overlap scheduler；不能与 EAGLE 共存 → 二选一 A/B |
| P2 | **EAGLE3** + 域内 draft | 需训 | accept 4~6+，上限最高 | 需 EAGLE3 头或 SpecForge 训；可喂自家 rollout 做域内适配 |
| ✗ | STANDALONE（小 draft） | — | 通常打不过 EAGLE | 跳过 |

### P0-a：EAGLE tree 扫参（零训练，先做）

同一个 EAGLE 头直接支持 tree。当前 `topk=1` 是最弱配置。建议扫：

| topk | num-steps | num-draft-tokens |
|-----:|----------:|-----------------:|
| 4    | 4         | 16               |
| 8    | 5         | 32               |

### P0-b：NGRAM（最贴 workload，零训练）

多轮编辑中第 2/3 轮大段复刻自己写过的 kernel —— suffix/ngram 主场。NGRAM 从 prompt + 已生成 token 建后缀 trie，复刻段单次可接受 10~18 token。

<!-- SuffixDecoding（NeurIPS'25）在 agentic/自相似 workload 实测比 EAGLE-2/3 快 2.8×。 -->

启用方式（无 draft model，CUDA-only）：

```bash
--sglang-speculative-algorithm NGRAM
--speculative-ngram-max-trie-depth 18
--speculative-ngram-max-bfs-breadth 10
--speculative-ngram-match-type BFS
```

**约束**：需关 overlap scheduler、不能与 EAGLE 共存 → 与 EAGLE 二选一 A/B。第 1 轮纯新生成 NGRAM 几乎不接受（EAGLE 赢），第 2/3 轮复刻段 NGRAM 碾压。**先量 turn-2+ 输出里"从上下文复制"占比**，占比高就倾向 NGRAM。

### P2：EAGLE3 + 域内 draft

<!-- EAGLE3 在 SGLang 里 accept 更高、吞吐最好。需 Qwen3.6-27B 的 EAGLE3 draft 头（先查官方，否则用 SpecForge 训）。加分：训 draft 时喂自家 kernel RL rollout 做域内适配，把当前 0.7 的 accept rate 顶上去。上限最高但要训练投入，放最后。 -->

需 Qwen3.6-27B 的 EAGLE3 draft 头（先查官方，否则用 SpecForge 训）。上限最高但需训练投入，放最后。

---

## 3. 下一步

1. **P0**：扫 EAGLE tree 参数 + 起一个 NGRAM A/B run。
2. 判优在 **纯 decode per-token 口径**（sglang `ignore_eos` bench）比 EAGLE-tree vs NGRAM，别只看 wall（wall 被 KernelGym 摊薄、区分度低）。
3. 不够再上 P2 训域内 EAGLE3 头。

---

## 4. 参考

- [SGLang spec decoding 文档](https://sgl-project.github.io/advanced_features/speculative_decoding.html)
- [SuffixDecoding（NeurIPS'25）](https://suffix-decoding.github.io/)
- [SpecForge（训 draft 头）](https://www.lmsys.org/blog/2025-07-25-spec-forge/)
- [Hybrid Models Meet SGLang](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)
