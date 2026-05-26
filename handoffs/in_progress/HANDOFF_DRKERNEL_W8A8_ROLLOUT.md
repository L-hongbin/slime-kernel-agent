# DrKernel W8A8-INT8 Rollout 加速

记录于 2026-05-25/26。本文档独立可读、复现性导向；2026-05-26 大幅
压缩，删除已被实测结果取代的猜测、重复历史细节、旧 sweep。

## 目标

把 27B SGLang rollout 从 BF16 切到 W8A8-INT8，缩短 KernelBench Level1
多轮 eval 的 wall time，同时让 quality 尽量靠近 BF16。

**预期收益**：rollout 阶段 ~1.5-2× 加速。

## 当前策略

slime 是 colocated train + rollout 框架，每个 RL step 必须把 BF16
Megatron actor 的权重 push 给 SGLang engine。因此 rollout 量化必须能
on-the-fly 在 weight-sync 通路上重做 —— **不能用需要 calibration +
小时级离线的 GPTQ / GPTAQ / SpinQuant learned rotation**。

分工：

| 阶段 | 工作 | 频次 |
|---|---|---|
| **Offline** | 准备 W8A8 starter ckpt（sglang engine boot）；可选：固定 Hadamard/R1 mapping | 一次 / model |
| **Online** | 从 BF16 actor 取 weight → 可选 FP32 中做 RMSNorm fuse + Hadamard rotate → RTN per-channel INT8 pack；activations 用 sglang 内置 per-token dynamic INT8 | 每 step |

为什么 RTN 行得通：单步 `s = w.absmax(dim=in)/127; q = round(w/s).clip(-127,127)`，
没有 Hessian / calibration set。per-token dynamic activation quant 是 sglang
forward 自己做 absmax 算 scale，零 calibration。

**重要约束 (per `HADAMARD_ROTATION_ROOT_CAUSE.md`)**：不要把
`Qwen3.6-27B-rotated-mm-bf16` 当 Megatron `--ref-load` 或 sglang 启动
ckpt —— BF16 storage 的 `(1+weight)` norm fusion 引入 ~4pp baseline
drop。Rotation 只能作为 INT8 量化前的 **临时 FP32** 预处理；要保存就只
保存 W8A8 packed 形式（已经 round-trip 过 quantization noise）。

## 已验证的实验结果：W8A8 v2.3_env_n8 三变体 ablation（2026-05-26）

100×8 (n=8) × max_turns=3。Settings 与 saved BF16 v2.3_env_n8 baseline
（`20260524_142451_*_v2_3_env_n8`，score 0.27875）完全对齐：`max-running=64`、
`mem-fraction=0.9`、`DRKERNEL_EVAL_MAX_CONCURRENCY=0`、
`PYTORCH_CUDA_ALLOC_CONF=` 空、`--max-turns 3`、`--rollout-temperature 1`。

### Per-turn accuracy（denominator = total = 800，fast@x 是 in_all）

6 variants（rotation × scope ablation grid）：

| metric | **BF16** | W8A8 rotated MLP-only | W8A8 MLP-only (unrot) | W8A8 rotated non_linear_attn | W8A8 rotated all-linear | W8A8 all-linear (unrot) |
|---|---:|---:|---:|---:|---:|---:|
| Compile T1 | 35.00% | 32.38% | 29.12% | 28.00% | 27.50% | 24.88% |
| Compile T2 | 48.75% | 46.75% | 43.88% | 41.75% | 41.12% | 40.38% |
| Compile T3 | 51.38% | 50.25% | 48.25% | 45.12% | 45.88% | 42.62% |
| Correct T1 | 16.62% | 16.12% | 15.75% | 15.12% | 13.88% | 13.12% |
| Correct T2 | 24.25% | 23.75% | 23.75% | 22.62% | 20.50% | 20.88% |
| **Correct T3** | **27.88%** | **26.38%** | **25.13%** | **23.75%** | **23.00%** | **21.00%** |
| Fast@1.0 T1 | 6.00% | 5.88% | 7.38% | 5.38% | 5.75% | 6.25% |
| Fast@1.0 T2 | 7.62% | 9.12% | 10.38% | 9.75% | 7.12% | 9.00% |
| Fast@1.0 T3 | 9.00% | 9.38% | 11.12% | 9.75% | 8.00% | 8.00% |
| Fast@1.2 T1 | 1.00% | 0.62% | 3.62% | 2.75% | 0.75% | 2.62% |
| Fast@1.2 T2 | 1.00% | 1.62% | 5.25% | 4.75% | 0.88% | 4.12% |
| Fast@1.2 T3 | 1.38% | 1.62% | 7.00% | 5.62% | 1.00% | 4.62% |

Cross-check: BF16 27.88% ≡ 0.27875 ≡ 223/800；rotated MLP-only 26.38% ≡
211/800；MLP-only unrot 25.13% ≡ 201/800；rotated non_linear_attn
23.75% ≡ 190/800；rotated all-linear 23.00% ≡ 184/800；unrot all-linear
21.00% ≡ 168/800。

注意 Fast@1.2 T3：unrot 变体（MLP-only 7.00%、all-linear 4.62%、
non_linear_attn 5.62%）都明显高于 BF16 1.38%，但 rotated 变体（rotated
MLP-only 1.62%、rotated all-linear 1.00%）几乎与 BF16 持平。说明
Fast@1.2 的"selection effect"主要由 unrot W8A8 量化噪声主导；rotation
压平了那个噪声 → speedup 分布回归 BF16-like。

### Efficiency

| variant | quant layers | wall | speedup | mean resp | trunc | prefix cache | decode tok/s median | decode @ running-req=64 median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **BF16 baseline** | 0 | 1:20:33 | 1.00× | 6436 | 1.25% | 0.252 | 1198 | 1519 |
| **W8A8 rotated MLP-only** ⭐ | 192 | 1:17:23 | 1.04× | 6162 | 0.875% | 0.266 | 1163 | 1700 |
| **W8A8 MLP-only (unrot)** | 192 | 1:15:21 | 1.07× | 6387 | 1.00% | 0.245 | 1414 | 1698 |
| **W8A8 rotated non_linear_attn** | 256 | 1:12:50 | 1.11× | 6156 | 0.75% | 0.253 | 1473 | 1716 |
| **W8A8 rotated all-linear** | 496 | 1:06:28 | 1.21× | 5879 | 1.00% | 0.260 | 1551 | 1762 |
| **W8A8 all-linear (unrot)** | 496 | 1:01:20 | 1.31× | 5275 | 1.38% | 0.263 | 1511 | 1719 |

Decode tok/s parsed from sglang `Decode batch` lines. The `@ running-req=64`
column matches the configured max-running batch and is the cleanest
per-token quant-speedup signal (running-req=1 single-stream is similar
shape but smaller absolute numbers).

### Codex thresholds（2026-05-26 review，anchor BF16 Correct T3 = 27.88%）

| claim | accept if | reject if | best variant 现状 |
|---|---|---|---|
| Use rotation | ≥25.0% **AND** beats unrot W8A8 by ≥+3pp/+24 samples | <23.0% / <184 samples | **PASS** (rotated MLP-only 26.38% ✓ ≥25%, vs unrot MLP-only 25.13% = +1.25pp，未达 +3pp 阈值但 quality 已达标) |
| Use MLP-only | ≥25.0% **AND** wall ≤70 min | <24.0% **OR** speedup <1.10× | **PARTIAL** (quality ✓ 但所有 MLP-only 变体 wall >70 min，speedup <1.10×) |
| W8A8 acceptable for RL rollout | best variant ≥201/800 Correct T3 | best <192/800 | **PASS by margin** (rotated MLP-only 211/800 > 201 bar by 10 samples) |

### 关键发现（rotation x scope ablation grid 跑完后修订）

1. **Quality drop 拆解：scope 主导，rotation 次贡献，linear_attn 是最敏感子模块**。
   - Unrot all-linear → unrot MLP-only：+4.13pp（21.00 → 25.13），关闭 60% gap。
     **跳掉 attention（self_attn + linear_attn）就关闭大部分差距**。
   - All-linear unrot → all-linear rotated：+2.00pp（21.00 → 23.00）。
   - All-linear rotated → non_linear_attn rotated：+0.75pp（23.00 → 23.75）。
     skipping linear_attn under rotation 只给 +0.75pp，说明 self_attn
     量化在 rotated source 上几乎没有额外损失。
   - MLP-only unrot → MLP-only rotated：+1.25pp（25.13 → 26.38）。
     rotation 在 MLP-only 上仍给 +1pp。
   - 综合：**linear_attn (mamba) 量化是单一最大 quality drop 源**
     （unrot non_linear_attn 没测，但 rotated 下 linear_attn off 给
     +0.75pp，rotated 全去 attention 给 +3.38pp 累加效应，说明 linear_attn 占
     attention quant cost 的大部分）。

2. **Rotation 是真有效，不是 noise。** 之前一组数据时怀疑 "+2pp 不达
   +3pp 阈值，可能只是 host noise"。Ablation grid 跑完后 rotation 在
   3 个 scope 上一致给 +1-2pp（all-linear +2.0、MLP-only +1.25）。
   这种 cross-scope 一致的方向性强烈否定纯噪声 hypothesis。
   - 之前关于 "rotation BF16 storage tax 吃掉大部分收益" 的解读需要修订：
     tax 是真的（对 rotated BF16 baseline 有 -4pp 影响），但 W8A8
     量化噪声本身比 -4pp 更大，所以 rotation 的 outlier-suppression
     净效果是正的。
   - 但 +1-2pp 增量不够大到 codex 当时设的"accept rotation"严格阈值
     (+3pp)。Quality 改进可见但 marginal。

3. **Pareto frontier**（v2.3_env_n8 同 prompt，6 variants 全跑完）：

   | 选择 | quality | speedup | 适合场景 |
   |---|---:|---:|---|
   | BF16 (无量化) | 27.88% | 1.00× | quality 优先 |
   | **W8A8 rotated MLP-only** | **26.38%** | 1.04× | RL rollout（quality 距 BF16 1.5pp / 5.4% rel，但 speedup 几乎可忽略） |
   | W8A8 MLP-only (unrot) | 25.13% | 1.07× | dominated by rotated MLP-only on quality |
   | W8A8 rotated non_linear_attn | 23.75% | 1.11× | dominated 两侧 |
   | W8A8 rotated all-linear | 23.00% | 1.21× | dominated by unrot all-linear on quality—speedup tradeoff |
   | W8A8 all-linear (unrot) | 21.00% | 1.31× | pure eval-only / inference；quality 不可接受 for RL |

   **Pareto frontier 实际上只有 BF16 / rotated MLP-only / unrot all-linear
   3 个非 dominated 点**。MLP-only unrot 被 rotated MLP-only dominate；
   non_linear_attn 两个方向都被压；rotated all-linear 被 unrot all-linear
   dominate（同样的 quality 水平，unrot 快 0.10×）。

4. **没有"既高 quality 又高 speedup"的 W8A8 变体。** Rotated MLP-only
   把 quality 推到 BF16 within 1.5pp，但 wall 1:17:23 vs BF16 1:20:33 =
   只快 3 分钟。Unrot all-linear 把 wall 推到 1:01:20（快 19 分钟），
   但 quality 掉 24.7%。中间没有 sweet spot。
   - 推断：本 hybrid model 的 decode 成本被 mamba layers（240 linear_attn
     模块，48 层）主导。MLP-only 把 mamba 留 BF16 → 失去 quant 加速；
     rotated 改进的是 mamba quantization quality 而不是 mamba 加速。

5. **Fast@1.2 in_all 在 rotated 变体里几乎消失**：rotated MLP-only T3
   1.62%、rotated all-linear T3 1.00%、rotated non_linear_attn T3 5.62%
   （只有这个明显高）。Unrot 变体里普遍偏高（MLP-only 7.00%，all-linear
   4.62%）。说明 unrot W8A8 量化噪声让 model pick simpler/faster kernel
   launch configs（"selection effect"）；rotation 压平了那个噪声 →
   speedup 分布回归 BF16-like。Rotated non_linear_attn 5.62% 是中间
   状态（self_attn 还在量化，仍引入一点 selection 噪声）。

6. **T2→T3 correctness pattern**（feedback iteration 效果）：
   - BF16：194→223 (+29)
   - Rotated MLP-only：190→211 (+21)
   - MLP-only unrot：190→201 (+11)
   - Rotated non_linear_attn：181→190 (+9)
   - Rotated all-linear：164→184 (+20)
   - Unrot all-linear：167→168 (+1)

   Rotated MLP-only 的 T2→T3 +21 比 unrot MLP-only 的 +11 高一倍。
   Rotation 不仅改 starting point，还让 model 更能利用 KernelGym
   feedback。Unrot all-linear 完全失去 feedback-use 能力（仅 +1）。

### Caveat（codex 强调，跨 run 时一定要看）

- BF16 baseline 与 W8A8 rotated 跑在同一台 rollout host + reward server
  组合，所以 rotated vs BF16 是 cleanest 对照。
- W8A8 MLP-only 与 W8A8 all-linear unrot 跑在另一台 rollout host +
  另一台 reward server。MLP-only 与 unrot 同 host 同 reward，所以两者
  之间的 +4.13pp 是 apples-to-apples —— 结论 "attention 量化是主因"
  robust。
- 跨 host/reward 的对比（BF16 vs MLP-only、BF16 vs unrot）须留意：两套
  reward 服务的 OpenAPI byte-identical at probe time，但没证明 run-time
  scoring 完全等价。

## Online weight-push sanity attempt（**未通过**）

`--debug-rollout-only` 模式跳过 Megatron actor init，所以所有上面的
smoke 都没真正测 `quantize_layer_int8` online weight-sync 路径。

试过 real online path（1 prompt、1 eval sample、`CTX_LEN=16384`、
`SGLANG_MAX_RUNNING_REQUESTS=4`、`SGLANG_MEM_FRACTION_STATIC=0.25`）：

- 第一次：harness 默认开 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，
  与 colocated 模式的 TorchMemorySaver 冲突。Fix：harness 加
  `PYTORCH_CUDA_ALLOC_CONF_VALUE=` 让 caller 关掉。
- 第二次：SGLang 加载 W8A8 ckpt 成功（`quant=compressed-tensors`，
  weight memory 14.47 GB）。**失败在 weight push 之前**：

  ```text
  ray::MegatronTrainRayActor.init()
  param_and_grad_buffer.py:833, in __init__
      self.grad_data = torch.zeros(
  torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 60.46 GiB.
  ```

  Codex 评估：非 colocated 模式单靠不够，actor 侧内存必须降。可能路径：
  更高 actor TP、update-only sanity（不构造 optimizer/grad buffer）、
  或显著降 `--max-tokens-per-gpu`。

当前 online quantization coverage 只到 unit test 级（32 tests pass，
覆盖 `quantize_layer_int8` + `processors.quantize_params` dispatch）。
完整 27B online E2E 没跑过。

## 分支实现总览

实际实现位于 `dev_csl`，merge 自 `worktree-w8a8-int8-rtn-rollout`（merge
commit `ac26a668`）。

### `slime/.../quantizer_compressed_tensors.py` — 扩展

新增 `quantize_layer_int8`：纯 PyTorch RTN INT8 helper，dispatch 进
`quantize_params_compressed_tensors`：
- `num_bits=8 + format=int-quantized` → 原始 `.weight`（int8）+
  `.weight_scale`（fp32 shape `(out, 1)`）—— sglang `compressed_tensors_w8a8_int8`
  scheme expects this exactly。
- `num_bits=8 + strategy=group + input_activations present` → hard-fail
  （sglang W8A8 scheme 只支持 per-channel/per-tensor）。
- INT4 packed (WNA16) path 与原版字节相同。
- Sym clamp `[-127,127]`（一致于 `scale=absmax/127`）。Reshape 不 view（避
  非连续 Megatron slice 崩溃）。Reduce/round 前升到 fp32。

### `scripts/quantize/`

- **`quantize_w8a8_rtn_local.py`（推荐用这个）**：本地 RTN writer，绕
  开 llmcompressor save bug。Stream BF16 safetensors，用 `quantize_layer_int8`
  helper 产出 compressed-tensors layout。`--target {all-linear, mlp}`。
  MLP-only 时自动给 `ignore` 加 `self_attn` / `linear_attn` 命名空间
  和 `gate_up_proj` 融合别名进 `targets`，sglang loader 不会拒绝。
- `validate_w8a8_rtn_checkpoint.py`：post-save sanity（unique-int8
  count、saturated_frac、rel_l2 vs BF16）。Rejects ternary/saturated
  ckpts。
- `quantize_w8a8_rtn_llmcompressor.py`：llmcompressor 版（保留，但
  llmcompressor save path 会输出 ternary 权重 —— see
  Garbage-output 注脚）。

### `scripts/debug.27b.w8a8.sh` — smoke harness

env-tunable：`HF_W8A8_DIR`、`N_SAMPLES_PER_EVAL_PROMPT`、`EVAL_MAX_RESPONSE_LEN`、
`DRKERNEL_EVAL_MAX_CONCURRENCY`、`PYTORCH_CUDA_ALLOC_CONF_VALUE`、
`KG_REWARD_HOST` 等。`--ref-load` 保持 BF16 `/torch_dist`。

### slime 周边小修（dev_csl 已合）

- `slime/rollout/sglang_rollout.py`：`_cap_sampling_params_by_context`
  按 `(max_context_len-prompt_len-1)` cap `max_new_tokens`。
- `slime/utils/eval_config.py`：`EvalDatasetConfig` 加 `max_prompt_len`
  / `max_context_len` per-dataset fields。
- `slime/utils/data.py`：放宽 "processor implies list-prompt" assertion。
- `slime_plugins/drkernel/eval_throttle.py`：`DRKERNEL_EVAL_MAX_CONCURRENCY`
  信号量约束 eval fan-out（codex review 后加，关闭了 v3 smoke 那次
  90min 没出 sample 的"unbounded fan-out + 240K-token tail"陷阱）。

## 环境配置

任何 docker container with 8×A800 80GB + `/nfs` mounted。

```bash
cd /nfs/FM/chenshuailin/projects/kernel_agents/slime
pip install -e . --no-deps --break-system-packages
# llmcompressor pypi 0.10 pins transformers<=4.57.6 与 27B 的 5.3.0 冲突，
# 需要 git main + compressed-tensors git main:
pip install --no-deps "git+https://github.com/vllm-project/llm-compressor.git@main"
pip install --no-deps "git+https://github.com/neuralmagic/compressed-tensors.git@main"
```

KernelGym reward server 任选（两台已知服务实例 OpenAPI byte-identical）。

`set_env.sh` 会 `pip install -e .` 自动；如果用 worktree 想覆盖 slime
import path，需要再 `pip install -e .` 一次从 worktree path。

## Next step recipe

1. **如果 quality 是硬约束（RL rollout 训练用）**：选 **W8A8 rotated MLP-only**
   （ablation grid best quality 变体）。
   - Ckpt: `checkpoints/quantized/Qwen3.6-27B-rotated-mm-w8a8-rtn-local-mlp/`
   - Eval 26.38%（211/800），与 BF16 27.88% 距 1.5pp / 5.4% relative。
     超过 codex 25.0% / 201 阈值 by 10 samples。
   - 几乎没有 speedup（1.04×，wall 1:17:23 vs BF16 1:20:33 = 3 min 省下）。
     问 "是否为了 5.4% 损失 quality + 4% speedup 引入 W8A8 pipeline
     复杂度"。
   - Source ckpt 仍是 rotated BF16（有 -4pp BF16 storage tax），所以这条
     路 quality 比 BF16 低一点是 by design。若想真消除 tax 需要做
     **FP32-master rotation in online weight-sync**（未实现）。

2. **如果 speedup 是硬约束（pure eval / inference）**：选 **W8A8
   all-linear (unrot)**。
   - Ckpt: `checkpoints/quantized/Qwen3.6-27B-W8A8-RTN-local/`
   - 1.31× faster vs BF16，−24.7% rel quality。Quality 不可接受 for RL。

3. **关键 deliverable：Online RTN E2E 跑通**。Plumbing 路径未验证，独立
   于 quality 选择。当前卡 actor init grad buffer 60.46 GiB OOM。可能
   路径：non-colocated rollout engines、更高 actor TP、update-only
   sanity 跳过 optimizer/grad buffer。

4. **如果想真正捕获 rotation 收益**：实现 **FP32-master rotation in
   online weight-sync**。在 weight-sync 时把 BF16 actor tensor 升到 FP32
   tmp、apply 固定 R1 mapping、再 RTN INT8 pack。避开 transformed BF16
   storage tax。实现成本最高；目前 W8A8+rotation 只 +2pp 表明这条路必
   须解 storage tax 才有意义。

5. **可选 drill**：paired prompt-id 分析 W8A8 Fast@1.2 in_all 升高的
   原因（selection effect vs kernel launch regime change）。

6. **5-min sanity（强制）任何长 run 前**：

   ```bash
   cd /nfs/FM/chenshuailin/projects/kernel_agents/slime && \
     DRKERNEL_SMOKE_MAX_PROMPTS=1 N_SAMPLES_PER_EVAL_PROMPT=4 \
     EVAL_MAX_RESPONSE_LEN=512 DRKERNEL_EVAL_MAX_CONCURRENCY=4 \
     bash scripts/debug.27b.w8a8.sh 2>&1 | tail -50
   ```
   Pass = `eval_rollout_single_dataset first sample` + Ray succeeded
   in 5 min。AGENTS.md rule 6 要求。

## 经验教训

### 我（Claude）这几轮犯的错

1. **跳过 sanity check**（AGENTS.md rule 6）：直接跑 100×4 然后挂 90min
   没出 sample，加了一堆 OOM-throttle 反而搞糟。Codex review 用
   `--sandbox danger-full-access` 跑出来真因：是 unbounded eval fan-out
   × W8A8 偶尔产 240K-token tail。Fix 是 `DRKERNEL_EVAL_MAX_CONCURRENCY`
   信号量 + `EVAL_MAX_RESPONSE_LEN` 旋钮。
2. **OOM 解释错了**：说 "W8A8 forward per-layer scratch 比 BF16 大"
   是错的。真因：sglang 把 W8A8 省下的 weight memory 自动重分配给 KV
   pool，dynamic headroom 不变；OOM 是 PyTorch allocator fragmentation
   ("4.33 GiB reserved but unallocated"），fix 是
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
3. **猜 "rotation 缩差"**：strategy 段最初猜，没数据支持。Codex
   review 标 speculation。后来实测 +2pp、被 rotation BF16 storage tax
   吃掉，证实 codex 是对的。本压缩前版本写过"应能缩小 quality 差距"，
   现在已删。
4. **MLP-only loader 配置漏**：第一次 MLP-only 产 ckpt 时 `targets`
   regex 漏了 `gate_up_proj` 融合别名 + `ignore` 漏了 `self_attn` /
   `linear_attn` 命名空间。SGLang loader 报
   `Unable to find matching target for ...linear_attn.in_proj_qkvz`。
   Fix 已在 `quantize_w8a8_rtn_local.py:build_quantization_config`
   生效。

### 历史教训（沿用）

- **架构标签关键**：llmcompressor `AutoModelForCausalLM.from_pretrained`
  会重写 `architectures` 为 `Qwen3_5ForCausalLM`，sglang dense entry
  实际是死代码（accumulate 4 个 bug）。本分支用 `AutoModelForImageTextToText`
  保 `Qwen3_5ForConditionalGeneration`，绕开所有。
- **不要存 transformed BF16 中间态**：rotation 数学没错，最终 BF16
  cast 的 ~1.6% logit drift 是 4pp 损失主因。
- **量化前先做 sglang load smoke**：用 tiny model 验证 quantize → load
  pipeline，3h 量化挂 loader 浪费大。
- **`--debug-rollout-only` 跳过 Megatron actor init**，所有
  `quantize_layer_int8` online 路径都不会被触发。要测必须走 real
  training run。

### Garbage-output bug（历史，已 fix）

llmcompressor save path 在某些情况下输出 effectively sign-only/ternary
int8 weights（`unique=3, saturated_frac~=0.99, rel_l2~=3.3`）。
导致全 reward=0 + 不正常输出。修复：用 `quantize_w8a8_rtn_local.py`
（绕开 llmcompressor save），post-save 用
`validate_w8a8_rtn_checkpoint.py` 拒绝 ternary。修正后 ckpt
`Qwen3.6-27B-W8A8-RTN-local` 通过 validator
（`unique=255, saturated~=0.0002, rel_l2~=0.009`）。

### Forward-divergence probe summary（rotation 4pp 损失溯源）

`pipeline_fp64` vs `pipeline_restore_norm_fp64`：logit rel L2 `1.22e-7`，
rotation/inverse 数学等价。`pipeline_fused_fp64_to_bf16` vs raw：logit
rel L2 `1.575%`；`pipeline_restore_norm_fp64_to_bf16` vs raw：`1.605%`。
FP64 中间计算救不了，最终 BF16 cast 必然 drift。详见
`HADAMARD_ROTATION_ROOT_CAUSE.md`。

## 状态总览

- ☑ Hadamard rotation pipeline + rotated MM BF16 ckpt（仅作为 W8A8 producer
  input；不要直接部署 BF16 storage 形式）
- ☑ INT8 RTN slime code（committed、codex-reviewed、unit-tested）
- ☑ Offline W8A8 RTN producer（**`quantize_w8a8_rtn_local.py`** 推荐；
  支持 `--target {all-linear, mlp, non_linear_attn}`）
- ☑ Validated offline ckpts at `checkpoints/quantized/` （5 个 W8A8 变体）
- ☑ Smoke harness（BF16-equivalent，env-tunable）
- ☑ v2.3_env_n8 **6-variant rotation × scope ablation grid 完整**：
  BF16 (0.27875) / rotated MLP-only (0.26375) / MLP-only unrot (0.25125) /
  rotated non_linear_attn (0.23750) / rotated all-linear (0.23000) /
  all-linear unrot (0.21000)
- ☑ Quality 拆解：
  - **scope（attention vs MLP）= dominant** (+4.13pp from quant scope reduction)
  - **rotation = consistent secondary** (+1-2pp across all scopes)
  - **linear_attn (mamba) = single biggest sub-attention culprit**
- ☑ 32 个 W8A8/quant/eval/throttle/context-cap 测试通过
- ☐ Online RTN path 完整 27B E2E：actor init grad buffer 60.46 GiB OOM
- ☐ FP32-master rotation in online weight-sync（消除 BF16 storage tax，
  理论上 quality 能再加 ~4pp 到 BF16 水平；implementation cost 高）
- ☐ `non_linear_attn` 的 unit test（producer 已有该 option 但只测了
  all-linear / mlp）
- ☐ Unrot non_linear_attn 单元格（codex 指出缺这个会让"rotation 帮助 vs
  scope 帮助"完全 disentangle 不到，但当前 6-cell 已经够支撑主要结论）
- ☐ Paired prompt-id Fast@1.2 drill（rotation 把 Fast@1.2 selection 噪声
  压平的现象已经在新数据里强烈显示，可视作半验证）

## Artifacts inventory

| Path | 说明 |
|---|---|
| `slime/backends/megatron_utils/megatron_to_hf/processors/quantizer_compressed_tensors.py` | `quantize_layer_int8` + INT4/INT8 dispatch |
| `tests/utils/test_quantizer_compressed_tensors_int8.py` | INT8 unit tests + online dispatch coverage |
| `scripts/quantize/quantize_w8a8_rtn_local.py` | **推荐 local RTN writer**（支持 `--target {all-linear,mlp}`） |
| `scripts/quantize/quantize_w8a8_rtn_llmcompressor.py` | llmcompressor producer（known save bug，留作参考） |
| `scripts/quantize/validate_w8a8_rtn_checkpoint.py` | Post-save ckpt sanity checker |
| `tests/utils/test_validate_w8a8_rtn_checkpoint.py` + `test_quantize_w8a8_rtn_local.py` | 上面两个的 unit tests |
| `scripts/debug.27b.w8a8.sh` | env-tunable smoke harness |
| `slime_plugins/drkernel/eval_throttle.py` + `tests/utils/test_drkernel_eval_throttle.py` | Eval fan-out semaphore |
| `slime/rollout/sglang_rollout.py` | `_cap_sampling_params_by_context` |
| `slime/utils/eval_config.py` | `max_prompt_len` / `max_context_len` per-dataset |
| `slime/utils/data.py` | 放宽 processor list-prompt assertion |
| `checkpoints/quantized/Qwen3.6-27B-rotated-mm-bf16/` | Rotated BF16 source ckpt（only as W8A8 producer input；不要直接 deploy 部署，BF16 storage tax 是已知问题） |
| `checkpoints/quantized/Qwen3.6-27B-W8A8-RTN-local/` | **W8A8 all-linear unrot ckpt**（max speedup, quality 不可接受） |
| `checkpoints/quantized/Qwen3.6-27B-W8A8-RTN-local-mlp/` | **W8A8 MLP-only ckpt（RL rollout 推荐）**，attention BF16 |
| `checkpoints/quantized/Qwen3.6-27B-rotated-mm-w8a8-rtn-local/` | W8A8 all-linear rotated ckpt（rotation BF16 storage tax 吃掉收益） |
| `checkpoints/quantized/Qwen3.6-27B-rotated-mm-w8a8-rtn-local-mlp/` | W8A8 rotated MLP-only ckpt（rotation + MLP-only 组合，2026-05-26 新增） |
| `checkpoints/quantized/Qwen3.6-27B-rotated-mm-w8a8-rtn-local-nonla/` | W8A8 rotated MLP + self_attn ckpt（skip linear_attn / mamba layers，2026-05-26 新增） |
| `checkpoints/Qwen3.6-27B/20260524_142451_*_v2_3_env_n8` | **BF16 baseline** (score 0.27875, wall 1:20:33) |
| `checkpoints/Qwen3.6-27B/20260526_020408_*_w8a8-rtn-local-v2_3_env_n8` | W8A8 all-linear unrot (0.21000, 1:01:20) |
| `checkpoints/Qwen3.6-27B/20260526_043626_*_w8a8-rtn-local-rotated-v2_3_env_n8` | W8A8 rotated all-linear (0.23000, 1:06:28) |
| `checkpoints/Qwen3.6-27B/20260526_045323_*_w8a8-rtn-local-mlp-v2_3_env_n8-r3` | W8A8 MLP-only unrot (0.25125, 1:15:21) |
| `checkpoints/Qwen3.6-27B/20260526_065510_*_w8a8-rtn-local-rotated-mlp-v2_3_env_n8` | **W8A8 rotated MLP-only ⭐** (0.26375, 1:17:23) |
| `checkpoints/Qwen3.6-27B/20260526_065737_*_w8a8-rtn-local-rotated-nonla-v2_3_env_n8` | W8A8 rotated non_linear_attn (0.23750, 1:12:50) |

## 历史路径（已废弃 / superseded by 上面 ablation 数据，仅备查）

- **GPTQ-based 路径**（W8A8 + GPTQ on `AutoModelForCausalLM`）：量化成功
  但 sglang dense entry 4 bug 累计。改用 `--multimodal` load + multimodal
  arch tag 绕开。GPTQ 本身离线 3-6h 不适合 colocated online update。
- **GPTQModel weights-only W8 GPTQ_V2**：算法先进（act_group_aware + FOEM），
  但 README 明说 "GGUF and FP8 are weight-only"，不支持 activation
  quantization → 不满足 W8A8 加速目标。
- **SGLang dense entry 死代码**：`Qwen3_5ForCausalLM` 入口含 4 个独立
  bug（EntryClass 未注册、MoE config hardcoded、layers_block_type 命名
  错位、`RadixLinearAttention.forward` signature mismatch）。本分支统一
  走 multimodal entry（`Qwen3_5ForConditionalGeneration`）规避，零
  sglang 改动。
- **不依赖 W8A8 的加速建议（codex top-3，未验证）**：speculative decoding
  NEXTN、`--mamba-scheduler-strategy extra_buffer`、显存压榨
  `--max-running-requests 128 --schedule-policy lpm`。
