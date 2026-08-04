# MusaCoder-27B 三轮评测复现（最终口径）

## 摘要

本文给出 MusaCoder-27B 在 KernelBench Level 1/2/3 上进行三轮 `load_inline` 实时反馈评测的当前复现方法。KernelGym 主分支尚未提供这条完整链路，因此必须同时使用已发布的 slime `feature/musacoder-multiturn` 分支和 KernelGym `feature/musacoder-load-inline` 分支。最终结果为：L1 T1 correct 86.50%、best-by-turn 96.50%；L2 为 68.00% / 92.50%；L3 为 24.50% / 54.50%。当前仍保留两个边界：服务端会拒绝超过 100KB 的原始响应；10800 秒 trajectory guard 只是绕开“超时取消会丢弃已完成轮次”的问题，数据保留根因尚未修复。

本文只覆盖最终三轮评测。单轮、旧 shape、binding 对比、one-shot ablation 和已作废的受污染运行均不在范围内。

## 1. 复现目标与指标

每个问题采样 8 条 trajectory，每条最多生成 3 轮。第 N+1 轮输入包含第 N 轮的编译、correctness、speedup 和错误反馈。模型始终输出一个完整 Python code block，其中使用 `torch.utils.cpp_extension.load_inline` 并定义 `ModelNew`。

所有 per-turn 指标和 best-by-turn 指标使用同一个分母：该 Level 的 trajectory 总数。

- `Compile@Tk`：第 k 轮完成编译的 trajectory 数 / trajectory 总数。
- `Correct@Tk`：第 k 轮 correctness 通过且未命中 decoy 检查的 trajectory 数 / trajectory 总数。
- `Best`：三轮中任意一轮满足对应条件的 trajectory 数 / trajectory 总数。
- `Fast@p`：correct 且 speedup ≥ p 的 trajectory 数 / trajectory 总数。

最终参考结果如下；斜杠内顺序均为 `T1 / T2 / T3 / best`。

| Level | trajectory | 产生的轮次 | Compile (%) | Correct (%) | Fast@1.0 (%) | Fast@1.2 (%) |
| --- | ---: | --- | --- | --- | --- | --- |
| L1 | 800 | 800 / 800 / 800 | 97.38 / 93.38 / 94.38 / 99.12 | **86.50 / 75.25 / 75.25 / 96.50** | 14.62 / 18.25 / 18.38 / 25.00 | 11.00 / 13.75 / 13.00 / 17.12 |
| L2 | 800 | 800 / 800 / 800 | 98.75 / 96.38 / 95.12 / 100.00 | **68.00 / 68.75 / 67.00 / 92.50** | 0.88 / 1.50 / 3.12 / 3.75 | 0.50 / 1.12 / 2.25 / 2.75 |
| L3 | 400 | 400 / 400 / 395 | 94.25 / 91.00 / 85.00 / 99.25 | **24.50 / 33.25 / 37.00 / 54.50** | 0.00 / 0.00 / 0.25 / 0.25 | 0.00 / 0.00 / 0.00 / 0.00 |

本次复核的原始汇总输出、数据集检查和测试结果保存在 [`local_artifacts/musacoder_mt3_reference_evidence.txt`](local_artifacts/musacoder_mt3_reference_evidence.txt)。

## 2. 固定版本与输入

### 2.1 代码版本

必须使用以下两个分支，并至少包含表中的固定代码 commit。KernelGym `main` 不能替代专用分支，它缺少 `Backend.LOAD_INLINE`、单块代码抽取、decoy 检查以及正确的 load_inline 路由。两个分支在固定代码 commit 之后可能还有纯文档提交，不影响复现代码基线。

| 组件 | 仓库与分支 | 固定代码 commit |
| --- | --- | --- |
| slime | [L-hongbin/slime-kernel-agent `feature/musacoder-multiturn`](https://github.com/L-hongbin/slime-kernel-agent/tree/feature/musacoder-multiturn) | `695664604b43b3b3c821e6c76fdb2cdad4b261e4` |
| KernelGym | [L-hongbin/KernelGYM `feature/musacoder-load-inline`](https://github.com/L-hongbin/KernelGYM/tree/feature/musacoder-load-inline) | `55018290b4c27d6f480a93eadac4de307d5b6c6d` |

共享环境已有对应 worktree：

```bash
SLIME_REPRO=/nfs/FM/chenshuailin/projects/kernel_agents/slime-musacoder-mt
KERNELGYM_REPRO=/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-load-inline

git -C "$SLIME_REPRO" status -sb
git -C "$SLIME_REPRO" rev-parse HEAD
git -C "$SLIME_REPRO" merge-base --is-ancestor \
  695664604b43b3b3c821e6c76fdb2cdad4b261e4 HEAD
git -C "$KERNELGYM_REPRO" status -sb
git -C "$KERNELGYM_REPRO" rev-parse HEAD
git -C "$KERNELGYM_REPRO" merge-base --is-ancestor \
  55018290b4c27d6f480a93eadac4de307d5b6c6d HEAD
```

期望两项祖先检查均返回 0。若要严格运行已验证的代码树，可将两个 worktree 分别 checkout 到表中的固定代码 commit。KernelGym worktree 中未跟踪的 `wheels` 是本地 wheel 目录链接，不属于分支内容。

### 2.2 模型和数据

```bash
MUSACODER_CKPT=/nfs/FM/chenshuailin/checkpoints/MooreThreads/MusaCoder-27B
MUSACODER_DATA_ROOT=/nfs/FM/chenshuailin/projects/kernel_agents/slime-dev-csl-2/Data

test -f "$MUSACODER_CKPT/config.json"
test -f "$MUSACODER_DATA_ROOT/kernelbench-level1-validation-musa-coder-load-inline/train.parquet"
test -f "$MUSACODER_DATA_ROOT/kernelbench-level2-validation-musa-coder-load-inline/train.parquet"
test -f "$MUSACODER_DATA_ROOT/kernelbench-level3-validation-musa-coder-load-inline/train.parquet"
```

三个 parquet 的预期行数依次为 100、100、50。每行 prompt 必须包含 `load_inline`，不得包含三段式标记 `### CUDA_KERNELS`；`reward_model.ground_truth` 必须非空。运行前执行：

```bash
python3 - <<'PY'
import pyarrow.parquet as pq

root = "/nfs/FM/chenshuailin/projects/kernel_agents/slime-dev-csl-2/Data"
for level, expected in ((1, 100), (2, 100), (3, 50)):
    path = f"{root}/kernelbench-level{level}-validation-musa-coder-load-inline/train.parquet"
    table = pq.read_table(path)
    assert table.num_rows == expected, (path, table.num_rows)
    row = table.slice(0, 1).to_pylist()[0]
    prompt = "\n".join(message["content"] for message in row["prompt"])
    assert "load_inline" in prompt
    assert "### CUDA_KERNELS" not in prompt
    assert row["reward_model"]["ground_truth"]
    print(f"L{level}: {table.num_rows} rows, prompt OK")
PY
```

如果运行环境缺少 `pyarrow`，先执行：

```bash
python3 -m pip install --target /tmp/musacoder_parquet_pkgs pyarrow jinja2
```

然后在运行上述检查脚本时给 `python3` 命令加上 `PYTHONPATH=/tmp/musacoder_parquet_pkgs`。

## 3. 运行拓扑

最终拓扑在 `.22` 单机的 8 张 A800-80G 上运行：

| 进程 | GPU | 代码 | 地址/并行度 |
| --- | --- | --- | --- |
| KernelGym reward | 0–3 | `feature/musacoder-load-inline` | `127.0.0.1:20111`；4 个 GPU worker；每 worker 一次只跑一个任务 |
| slime + SGLang rollout | 4–7 | `feature/musacoder-multiturn` | 1 个 TP=4 engine；`SGLANG_MAX_RUNNING_REQUESTS=32` |

登录入口：

```bash
ssh -p 24167 root@192.168.16.22
```

该端口可能随镜像升级改变；连接失败时先确认当前 `.22` SSH 端口。不要使用 `.21` 上的 KernelGym `main` 服务。

## 4. 启动并检查 KernelGym

`load_inline` 的编译和加载必须在同一个 worker 进程中完成。服务端和 slime 客户端都要设置 `split_compile_and_execute=false`，否则 PyTorch JIT extension versioner 可能让编译进程生成 `<name>_vN.so`，执行进程却加载另一个版本名，形成假的 `cannot open shared object` 编译失败。

首次准备环境：

```bash
cd /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-load-inline
bash ensure_venv.sh
uv pip install --python .venv/bin/python einops==0.8.2

.venv/bin/python - <<'PY'
from kernelgym.common import Backend
from kernelgym.deployment_profiles import get_profile

env = get_profile("v1").env()
assert Backend.LOAD_INLINE.value == "load_inline"
assert env["SPLIT_COMPILE_AND_EXECUTE"] == "false"
assert env["GPU_DEVICES"] == "[0,1,2,3]"
print("KernelGym load_inline profile OK")
PY

PYTHONPATH=/nfs/FM/chenshuailin/projects/kernel_agents/slime-musacoder-mt python3 - <<'PY'
from examples.kernel_agent.config import CUDA_AGENT_CONFIGS

env = CUDA_AGENT_CONFIGS["env"]
assert env["split_compile_and_execute"] is False
assert env["kernel_eval_task_timeout"] == 600
assert env["kernel_eval_client_timeout"] == 4800
assert env["kernel_eval_acquire_timeout"] == 4800
print("slime load_inline timeout/split config OK")
PY
```

`einops==0.8.2` 是 L3 Mamba2 reference 的运行依赖。缺失时 reference 加载失败，旧代码路径会把真实 import 错误掩盖成 `cannot unpack non-iterable NoneType`。

在单独的 tmux pane 中启动 reward 服务：

```bash
cd /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-load-inline
CUDA_VISIBLE_DEVICES=0,1,2,3 \
KERNELGYM_CORRECTNESS_DISABLE_TF32=1 \
bash deploy_node.sh --nnodes 1
```

启动或重启服务前先确认 GPU 0–3 和端口 20111 没有被其他任务使用。另一个 pane 中检查：

```bash
cd /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-load-inline
bash check_node.sh
bash test_reward.sh
curl -fsS http://127.0.0.1:20111/health
```

correctness 使用 fp32 tolerance `1e-4`，并关闭 TF32；profile 阶段恢复默认性能策略。KernelBench reference 按 train mode 运行，因此 BatchNorm 使用当前 batch 统计量。

## 5. 启动三轮评测

L1、L2、L3 必须串行执行，它们共享 GPU 4–7、Ray 端口和临时目录。启动脚本会执行 `ray stop --force`，所以运行前必须确认 `.22` 上没有需要保留的 Ray 作业。脚本未启用 eager mode。

先设置公共参数：

```bash
cd /nfs/FM/chenshuailin/projects/kernel_agents/slime-musacoder-mt

export CUDA_VISIBLE_DEVICES=4,5,6,7
export NUM_GPUS=4
export ACTOR_NUM_GPUS=4
export EVAL_HF_CKPT=/nfs/FM/chenshuailin/checkpoints/MooreThreads/MusaCoder-27B
export KERNEL_ENV_URL=http://127.0.0.1:20111
export KERNEL_BACKEND=load_inline
export MAX_TURNS=3
export N_SAMPLES_PER_EVAL_PROMPT=8
export EVAL_TEMPERATURE=0.7
export EVAL_TOP_P=0.95
export MAX_CONTEXT_LEN=40960
export MAX_RESPONSE_LEN=32768
export SGLANG_MAX_RUNNING_REQUESTS=32
export KERNEL_AGENT_GENERATE_GUARD_SEC=10800
export MASTER_ADDR=192.168.16.22
export LOCAL_GLOO_SOCKET_IFNAME=ens22f0np0
export RAY_PORT=6382
export RAY_DASHBOARD_PORT=8268
export RAY_TEMP_DIR=/tmp/ray_eval
export MUSACODER_RUN_STAMP=$(date +%Y%m%d.%H%M%S)
```

依次选择一个 Level 运行。每次命令完成并检查结果后，再启动下一个 Level。

L1：

```bash
export EVAL_DATA=/nfs/FM/chenshuailin/projects/kernel_agents/slime-dev-csl-2/Data/kernelbench-level1-validation-musa-coder-load-inline/train.parquet
export EVAL_TAG=MusaCoder-27B.mt3_level1_t600.repro.${MUSACODER_RUN_STAMP}
bash examples/kernel_agent/eval.mt3.musacoder.27B.sh
```

L2：

```bash
export EVAL_DATA=/nfs/FM/chenshuailin/projects/kernel_agents/slime-dev-csl-2/Data/kernelbench-level2-validation-musa-coder-load-inline/train.parquet
export EVAL_TAG=MusaCoder-27B.mt3_level2_t600.repro.${MUSACODER_RUN_STAMP}
bash examples/kernel_agent/eval.mt3.musacoder.27B.sh
```

L3：

```bash
export EVAL_DATA=/nfs/FM/chenshuailin/projects/kernel_agents/slime-dev-csl-2/Data/kernelbench-level3-validation-musa-coder-load-inline/train.parquet
export EVAL_TAG=MusaCoder-27B.mt3_level3_t600.repro.${MUSACODER_RUN_STAMP}
bash examples/kernel_agent/eval.mt3.musacoder.27B.sh
```

脚本运行的是 `--debug-rollout-only`，不启动训练 backend。关键运行口径已固定在代码和上述环境变量中：每题 8 个样本、temperature 0.7、top-p 0.95、3 轮、32K 最大回复、40K 总上下文、600 秒单次 reward task timeout、4800 秒 client/acquire timeout、10800 秒 trajectory guard。

## 6. 监控与完成判据

脚本开始后会打印实际 log 和 dump 目录。长任务期间同时检查进度、reward 服务和错误签名：

```bash
tail -f <EVAL_DIR>/<timestamp>.log
curl -fsS http://127.0.0.1:20111/health
nvidia-smi
rg -n "Traceback|cannot open shared object|generate guard|HTTP 400|timed out" <EVAL_DIR>/<timestamp>.log
```

不要以 Ray job 已提交、模型服务可访问或进度通知作为完成证据。完成至少满足：

1. `eval_0.pt` 已写入 `<EVAL_DIR>/dumps/rollout_data/`。
2. 汇总中的 `missing env_result` 为 0。
3. trajectory 数为 L1=800、L2=800、L3=400。
4. L1/L2 应各有 2400 条 turn record；L3 最多 1200 条，最终参考 artifact 为 1195 条，所有 per-turn 比率仍以 400 为分母。
5. 日志中没有 `cannot open shared object`；出现该签名说明 split 配置或进程隔离失效。
6. 抽查至少一个正确样本和一个失败样本的 response、feedback 与 `metadata.env_result`，确认下一轮确实收到上一轮的结构化结果。

汇总新运行：

```bash
python3 examples/kernel_agent/summarize_eval.py \
  <EVAL_DIR>/dumps/rollout_data/eval_0.pt \
  --max-turns 3
```

`summarize_eval.py` 必须来自 slime commit `69566460` 或之后；该版本会优先读取修复 dump 中的 `metadata.group_id`，避免把 L1 的 800 条 trajectory 错聚合成 662 条。

## 7. 复算现有最终 artifact

以下文件是当前定稿结果的权威输入：

```bash
MT3_ROOT=/nfs/FM/chenshuailin/projects/kernel_agents/slime-musacoder-mt/experiments/EvalMT3.load_inline.MusaCoder-27B.CTX40960

# L1：使用已替换假 cannot-open trajectory、且 metadata group/index 已规范化的最终文件。
python3 examples/kernel_agent/summarize_eval.py \
  "$MT3_ROOT/MusaCoder-27B.mt3_largeshape_nosplit.load_inline/dumps/rollout_data/eval_0_merged_clean.pt" \
  --max-turns 3

# L2：600s 最终运行。
python3 examples/kernel_agent/summarize_eval.py \
  "$MT3_ROOT/MusaCoder-27B.mt3_level2_t600.load_inline/dumps/rollout_data/eval_0.pt" \
  --max-turns 3

# L3：600s 最终运行。
python3 examples/kernel_agent/summarize_eval.py \
  "$MT3_ROOT/MusaCoder-27B.mt3_level3_t600.load_inline/dumps/rollout_data/eval_0.pt" \
  --max-turns 3
```

L1 同目录的普通 `eval_0.pt` 不代表定稿结果，不应进入结果表。

## 8. 结果边界

- 生成使用 temperature 0.7，SGLang 调度也会影响采样顺序。重新运行应验证口径、样本数和结果区间，不能要求逐 token 或百分比逐位相同。
- 当前发布分支把 reward task timeout 固定为 600 秒。L2/L3 参考结果就是这一口径。L1 表中定稿 artifact 使用 300 秒 task timeout，并替换了受 load_inline versioner 假失败影响的 179 条 trajectory；尚无全量 L1-600s 结果。用第 5 节命令重新跑 L1 会产生一个新的 600s endpoint，不能把它与表中的 L1 当成同一次测量。
- 服务端仍按原始 response 的 100KB 大小做校验。L3 的退化长回复可能收到 HTTP 400；为了保持结果可比，不要在复现时私自修改阈值或改为只检查抽取后的代码。
- `KERNEL_AGENT_GENERATE_GUARD_SEC=10800` 降低了 trajectory 被取消的概率。它没有修复取消路径丢弃已完成 T1/T2 record 的根因；若 guard 仍触发，必须把该运行标记为不完整并检查 dump，不能直接报告成功。
- `split_compile_and_execute=false` 会牺牲 CPU 编译流水线并增加总耗时，但它是当前避免 load_inline 假编译失败的已验证配置。
