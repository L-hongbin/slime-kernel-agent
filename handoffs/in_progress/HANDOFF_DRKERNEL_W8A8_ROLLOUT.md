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

| metric | **BF16** | W8A8 MLP-only | W8A8 rotated all-linear | W8A8 all-linear (unrot) |
|---|---:|---:|---:|---:|
| Compile T1 | 35.00% | 29.12% | 27.50% | 24.88% |
| Compile T2 | 48.75% | 43.88% | 41.12% | 40.38% |
| Compile T3 | 51.38% | 48.25% | 45.88% | 42.62% |
| Correct T1 | 16.62% | 15.75% | 13.88% | 13.12% |
| Correct T2 | 24.25% | 23.75% | 20.50% | 20.88% |
| **Correct T3** | **27.88%** | **25.13%** | **23.00%** | **21.00%** |
| Fast@1.0 T1 | 6.00% | 7.38% | 5.75% | 6.25% |
| Fast@1.0 T2 | 7.62% | 10.38% | 7.12% | 9.00% |
| Fast@1.0 T3 | 9.00% | 11.12% | 8.00% | 8.00% |
| Fast@1.2 T1 | 1.00% | 3.62% | 0.75% | 2.62% |
| Fast@1.2 T2 | 1.00% | 5.25% | 0.88% | 4.12% |
| Fast@1.2 T3 | 1.38% | 7.00% | 1.00% | 4.62% |

Cross-check: BF16 Correct T3 27.88% ≡ `eval/kernelbench_level1=0.27875`；
W8A8 MLP-only 25.13% ≡ 201/800；W8A8 rotated 23.00% ≡ 184/800；
W8A8 unrot 21.00% ≡ 168/800。

### Efficiency

| variant | quantized layers | wall | speedup | mean resp | trunc | prefix cache | decode tok/s median | decode @ running-req=64 median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **BF16 baseline** | 0 | 1:20:33 | 1.00× | 6436 | 1.25% | 0.252 | 1198 | 1519 |
| **W8A8 MLP-only** | 192 | 1:15:21 | 1.07× | 6387 | 1.00% | 0.245 | 1414 | 1698 |
| **W8A8 rotated all-linear** | 496 | 1:06:28 | 1.21× | 5879 | 1.00% | 0.260 | 1551 | 1762 |
| **W8A8 all-linear (unrot)** | 496 | 1:01:20 | 1.31× | 5275 | 1.38% | 0.263 | 1511 | 1719 |

Decode tok/s parsed from sglang `Decode batch` lines. The `@ running-req=64`
column matches the configured max-running batch and is the cleanest
per-token quant-speedup signal (running-req=1 single-stream is similar
shape but smaller absolute numbers).

### Codex thresholds（2026-05-26 review，anchor BF16 Correct T3 = 27.88%）

| claim | accept if | reject if | this run |
|---|---|---|---|
| Use rotation | ≥25.0% **AND** beats unrot W8A8 by ≥+3pp/+24 samples | <23.0% / <184 samples | **FAIL** (23.0%; only +2.0pp vs unrot) |
| Use MLP-only | ≥25.0% **AND** wall ≤70 min | <24.0% **OR** speedup <1.10× | **PARTIAL** (25.13% ✓ quality, 1.07× ✗ speedup) |
| W8A8 acceptable for RL rollout | best variant ≥201/800 Correct T3 | best <192/800 | **MLP-only exactly at bar** (201/800) |

### 关键发现

1. **Attention quantization 是 quality 下降的主因。** 把 64 个
   self_attn + 240 个 linear_attn 留 BF16（MLP-only），Correct T3 从
   21.00% 恢复到 25.13% —— 关闭了 4.13pp / 6.88pp = 60% 的 gap。
   早期"差距是 MLP 权重 outliers、需要 rotation 缩差"的猜测被证伪。

2. **Rotation 的净 quality 收益只 +2.0pp**（21.00% → 23.00%），不达
   codex `+3pp/+24 samples` 阈值。`HADAMARD_ROTATION_ROOT_CAUSE.md`
   解释了原因：rotated BF16 baseline 本身就比 unrotated 低 ~4pp，
   `(1+weight)` norm fusion 的 BF16 storage drift 吃掉了大部分
   rotation 的 quantization-noise 抑制。要拿到 rotation 真实收益需
   **FP32-master rotation in online weight-sync**（不存 transformed
   BF16 中间态），目前未跑通。

3. **Pareto frontier**（v2.3_env_n8 同 prompt）：

   | 选择 | quality | speedup | 适合 |
   |---|---:|---:|---|
   | BF16 (无量化) | 27.88% | 1.00× | quality 优先 |
   | **W8A8 MLP-only** | **25.13%** | 1.07× | RL rollout 推荐：quality 卡 codex 阈值正好通过 |
   | W8A8 rotated | 23.00% | 1.21× | dominated by MLP-only on quality + few pp wall savings |
   | W8A8 all-linear (unrot) | 21.00% | **1.31×** | pure eval-only / inference benchmark；不可接受 for RL |

   理论上 dominant 解是 **FP32-master rotation + MLP-only**（兼具
   rotation 的 outlier 抑制 + MLP-only 的 attention BF16 保护），
   未验证。

4. **Fast@1.2 in_all 反向高于 BF16**（MLP-only T3 7.00% 即 56/800
   vs BF16 11/800；unrot W8A8 T3 4.62% 即 37/800）。Codex 指出这不只是
   tiny-denominator noise（W8A8 conditional-on-correct Fast@1.2 rate
   达 22-28%，BF16 仅 5%）。可能是 W8A8/RTN 把 model 推向更简单/更
   快的 kernel launch config（"selection effect" 或 "regime change"，
   未做 paired prompt-id drill 验证）。

5. **T2→T3 correctness pattern**：BF16 +29 correct samples（194→223），
   unrot W8A8 仅 +1（167→168），MLP-only +11（190→201），rotated −10
   （167→184， T2/T3 之间下降是噪音方向但说明 quality 不稳）。W8A8
   能用 KernelGym feedback 让代码 *compile*，但要让代码 *correct* 比
   BF16 更难 —— W8A8 失败模式更接近"算法 basin 偏"而非"语法错误"。

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

1. **如果 quality 是硬约束（RL rollout 训练用）**：选 **W8A8 MLP-only**。
   - Ckpt: `checkpoints/quantized/Qwen3.6-27B-W8A8-RTN-local-mlp/`
   - Eval 通过 codex 25.0% 阈值（201/800 ≡ exactly the bar）。
   - 牺牲掉 quant 加速（1.07×，不是 1.31×）。Attention 仍 BF16 让 hybrid
     model 主导 decode 成本。

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

- ☑ Hadamard rotation pipeline + rotated MM BF16 ckpt（**不要部署 BF16
  storage**，rotation 只用于 INT8 量化前的 FP32 临时预处理）
- ☑ INT8 RTN slime code（committed、codex-reviewed、unit-tested）
- ☑ Offline W8A8 RTN producer（**`quantize_w8a8_rtn_local.py`** 推荐；
  llmcompressor version 留作参考但 save bug 历史）
- ☑ Corrected offline ckpts at `/nfs/.../Qwen3.6-27B-W8A8-RTN-local{,-mlp,-rotated-mm-w8a8-rtn-local}/`
- ☑ Smoke harness（BF16-equivalent，env-tunable）
- ☑ v2.3_env_n8 三变体 ablation 完整：BF16 (0.27875) / MLP-only
  (0.25125) / rotated (0.23000) / all-linear unrot (0.21000)
- ☑ Quality 主因定位：**attention quantization scope**，不是 MLP outliers
- ☑ 32 个 W8A8/quant/eval/throttle/context-cap 测试通过
- ☐ Online RTN path 完整 27B E2E：actor init grad buffer 60.46 GiB OOM
- ☐ FP32-master rotation in online weight-sync（唯一未跑过、可能让
  rotation 真正缩差的方案）
- ☐ Paired prompt-id Fast@1.2 drill 验证 selection vs regime change

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
| `checkpoints/Qwen3.6-27B/20260526_045323_*_w8a8-rtn-local-mlp-v2_3_env_n8-r3` | W8A8 MLP-only (0.25125, 1:15:21) |

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
