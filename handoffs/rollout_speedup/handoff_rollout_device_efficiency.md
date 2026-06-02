# Rollout 推理效率设备对比: A800 vs A100 vs H20

**注意**:A100 / A800 用 SGLang 的 **triton** linear-attn backend,而H20 用 **flashinfer**。
两边后端不同是**硬件限制**:flashinfer 的 linear-attn 后端不支持 A800。

而号称效率更高的 **flashqla 内核又尚未集成进 SGLang**

# A800 vs A100

**结论:A800 与 A100 在 C64 下推理效率与精度一致**：wall time 6608s vs 6421s(差 2.9%,噪声)，decode throughput 3995 vs 4105 tok/s(差 ~3%)，其他指标也同档。

| 指标                         |  A800 (C64) |  A100 (C64) |
| ---------------------------- | ----------: | ----------: |
| wall(800 eval)               |       6608s |       6421s |
| s/it                         |        8.26 |        8.03 |
| response_len/mean            |      8285.6 |      8606.3 |
| reward score                 |       0.375 |       0.375 |
| decode running-req median    |          22 |          26 |
| gen throughput mean / median | 1759 / 1738 | 1817 / 1882 |
| spec_accept_length           |      3.2367 |      3.2389 |
| prefix_cache_hit_rate        |      45.94% |      45.85% |


# A800 vs H20

**结论:精度一致,吞吐同档**


| 指标                         | A800 (C96) triton | H20 (C96) flashinfer |
| ---------------------------- | ----------------: | -------------------: |
| wall(800 eval)               |             5077s |            **4746s** |
| s/it                         |              6.35 |             **5.93** |
| response_len/mean            |            8366.2 |               7993.4 |
| reward score                 |             0.384 |                0.381 |
| decode running-req median    |            **90** |                   34 |
| gen throughput mean / median |   **2316 / 2915** |          2158 / 2470 |
| spec_accept_length           |            3.2390 |               3.2432 |
| prefix_cache_hit_rate        |            40.51% |               42.41% |


# A800 vs H20:reward 各阶段耗时

**结论:reward 各阶段耗时两边基本一致,且与 rollout 设备无关。** reward(KernelGym 评测)跑在**独立的 RTX 4090
worker 池**上,A800 / H20 只决定 rollout、不决定 reward,所以耗时本应相同;下表也证实如此。成本由 **performance
(性能测量)阶段主导**:正确 kernel 要在 4090 上跑 30 warmup + 50 计时 + profiling,单次 ~6–8s,远大于 compile
(~0.16s)与 correctness(~0.67s)。数据取自 dump 内每 turn 的 `kernelgym.metadata.kg_stage_completed_s`。

| reward 阶段                       | A800 (C96) mean / median | H20 (C96) mean / median | 口径 |
| -------------------------------- | -----------------------: | ----------------------: | ---- |
| compile_and_load                 |            0.171 / 0.164 |           0.161 / 0.157 | 所有评测 turn |
| correctness                      |            0.656 / 0.327 |           0.688 / 0.326 | 编译成功的 turn |
| **performance(性能测量)**        |          **7.63 / 3.90** |             6.43 / 2.45 | 仅正确 kernel |
| setup(load/prepare/build/detect) |            0.006 / 0.004 |           0.005 / 0.004 | 所有评测 turn |
| **每 turn 总 reward**            |          **4.78 / 0.31** |             4.59 / 0.37 | 所有 turn |

注:compile / correctness / 每 turn 总 reward 两边几乎相同(同一 4090 池)。performance 的 A800 7.63 vs H20 6.43
(median 3.90 vs 2.45)差异来自**被测 kernel 分布不同**(A800 这批正确 kernel 平均更慢、profiling 更久),不是 4090
reward 速度差。reward 与 decode 在 16 个 worker 上**并发**,此处是单 turn 的 reward 计算耗时,不直接等于 wall。

<!-- ## 下一步

1. **flashqla 一旦进 SGLang**,补 H20 flashqla run,测 linear-attn 解码吞吐上限——当前后端口径外最有潜力。
2. 严格设备 A/B:launcher 打印 `nvidia-smi --query-gpu=name` 与 driver / CUDA。要把第 2 章的"设备"与"backend"拆开,
   只能补一条 **H20 (C96, triton)**——**A800 跑不了 flashinfer linear-attn(不支持),无法在 A800 侧对齐**;当前两者
   绑定,A800 略快是设备 + triton 的合并效应。
3. 真正的 wall 杠杆是并发(C64→C96,−23%)与 reward overlap / specdec tree-NGRAM;换设备无明确收益。
4. A100 仍只有 C64,若要把 A100 也纳入 C96 口径需补 A100 (C96)。

## 证据

- 逐 turn / token 权威输出:`handoffs/in_progress/evidence_per_turn_quality.txt`(4 条 run 的 `eval_0.pt` 经
  `scripts/analysis/per_turn_acc_nopt.py` 解析)。
- A100 (C64): `checkpoints/Qwen3.6-27B/20260531_052414_...C64.A100/`(triton;`run.log` 已删,dump 仍在)
- A800 (C64): `checkpoints/Qwen3.6-27B/20260531_051603_...C64/`(triton;`run.log` 已删,dump 仍在)
- A800 (C96): `checkpoints/Qwen3.6-27B/20260529_132454_...rm16/run.log`(triton;score 0.38375,wall 1:24:37 / 6.35;
  reward-concurrency handoff §2 的 16-worker run)
- H20 (C96): `checkpoints/Qwen3.6-27B/20260602_042421_...C96.H20.linear-fi/run.log`(**flashinfer**;
  `linear_attn_backend='flashinfer'`,score 0.38125,wall 1:19:06 / 5.93,reach-T3 800/800,**0 KernelGymRequestError**)
  - 旧 H20 run 已弃用替换:`20260601_083536`(triton)与 `20260601_231948`(flashinfer)——后者撞上一次 KernelGym
    reward-server 瞬时故障(106 条 `KernelGymRequestError`,reach-T3 仅 747、score 0.355,口径不干净),已由本条干净 run 取代。 -->
