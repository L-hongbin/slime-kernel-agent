# Qwen3.5-9B 训练 Handoff

日期：2026-05-12

## 当前状态

训练已经停止。

- 机器：`chenshuailin@192.168.16.55`
- 容器：`slime-qwen9b-train-current`
- 镜像：`192.168.14.129:80/library/slime:nightly-dev-20260430b`
- 停止方式：`docker stop slime-qwen9b-train-current`
- 停止后容器状态：`status=exited exit=137 oom=false`
- 停止后 GPU 显存：8 张卡均为 `0 MiB`
- 当前分支：`dev_csl`

最新验证过的训练产物：

- 日志：`checkpoints/Qwen3.5-9B/run_20260511_231110.log`
- 最新 checkpoint marker：`checkpoints/Qwen3.5-9B/latest_checkpointed_iteration.txt`
- 停止时 marker 内容：`2`
- 已成功保存的 checkpoint：
  - `checkpoints/Qwen3.5-9B/iter_0000000`，约 `117G`
  - `checkpoints/Qwen3.5-9B/iter_0000001`，约 `117G`
  - `checkpoints/Qwen3.5-9B/iter_0000002`，约 `117G`
- 旧的失败半成品保留用于排查：
  - `checkpoints/Qwen3.5-9B/iter_0000019.failed_20260512_0410`，约 `49K`

## 入口脚本

当前训练入口：

```bash
scripts/run-qwen3.5-9B.sh
```

Ray 启动逻辑已拆到：

```bash
scripts/ray/start_cluster.sh
```

当前 Qwen3.5-9B 关键默认参数：

```bash
TP=2
--rollout-batch-size 32
--n-samples-per-prompt 8
--global-batch-size 256
--sglang-mem-fraction-static 0.7
SAVE_INTERVAL=1
```

当前训练参数仍是单机：

```bash
--actor-num-nodes 1
--actor-num-gpus-per-node ${NUM_GPUS}
```

## 问题 1：Ray raylet 启动失败，dashboard agent 端口文件超时

### 表现

Ray 启动时 raylet 直接 fatal，日志里反复出现：

```text
Timed out waiting for file .../dashboard_agent_listen_port_<node_id>
Timed out waiting for file .../metrics_export_port_<node_id>
Check failed: status...has_value()
```

另一个表现是 Ray CLI 超时：

```text
The current node timed out during startup.
GCS cannot find the node ...
RPC error: Deadline Exceeded
```

### 原因判断

这不是训练代码的问题，而是 Ray 在当前环境下 dashboard agent 启动较慢。raylet 只等很短时间就要求 dashboard agent 写出端口文件，端口文件还没落盘时 raylet 会 fatal。

同时 `/tmp/ray/session_*` 残留会干扰后续排查，旧日志和新日志混在一起。

### 解决方式

新增 `scripts/ray/start_cluster.sh`，把 Ray 启动从训练脚本里拆出来，并做了以下处理：

- 固定 Ray 关键端口：
  - `RAY_PORT=6379`
  - `RAY_DASHBOARD_PORT=8265`
  - `RAY_DASHBOARD_AGENT_GRPC_PORT=52366`
  - `RAY_DASHBOARD_AGENT_LISTEN_PORT=52365`
  - `RAY_RUNTIME_ENV_AGENT_PORT=52367`
  - `RAY_METRICS_EXPORT_PORT=20000`
- `RAY_METRICS_EXPORT_PORT=20000` 避开 Ray 默认 worker 端口段 `10002-19999`。
- 启动前清理旧进程：
  - `pkill -9 sglang`
  - `ray stop --force`
  - `pkill -9 ray`
  - `pkill -9 python`
- 默认清理旧 Ray session：
  - `/tmp/ray/session_*`
  - `/tmp/ray/session_latest`
- 增加 `RAY_raylet_start_wait_time_s=120`。
- 增加 `RAY_agent_register_timeout_ms=120000` 和 Ray `--system-config '{"agent_register_timeout_ms":120000}'`。
- 增加 `preseed_dashboard_agent_port_files`：
  - 从最新 `raylet.out` 里读取新 node id。
  - 主动写入 Ray 需要的 dashboard/metrics port 文件。
  - 避免 raylet 在 dashboard agent 真正 ready 前 fatal。
- 启动失败时自动 dump 关键 Ray 日志：
  - `raylet.err`
  - `raylet.out`
  - `dashboard_agent.log`
  - `dashboard_agent.err`
  - `gcs_server.*`

### 验证结果

后续 Ray head 可以稳定启动，日志出现：

```text
Ray runtime started.
Preseeded Ray dashboard port files for node ...
```

并且训练脚本可以继续进入 `ray job submit`。

## 问题 2：Ray job server 已启动但不能立即提交任务

### 表现

Ray runtime 已经起来，但提交训练任务时报：

```text
RuntimeError: Request failed with status code 500:
No available agent to submit job, please try again later.
```

### 原因判断

Ray dashboard/job server 和 dashboard agent ready 不是同一时刻。`ray start` 返回成功后，job agent 仍可能短时间不可用。

### 解决方式

在 `scripts/ray/start_cluster.sh` 里新增：

- `wait_for_ray_job_server`
  - 等 `dashboard_agent.log` 出现 `Dashboard agent http address:`
  - 再用 `ray job list --address=http://127.0.0.1:8265` 做最终可用性检查
- `submit_ray_job`
  - 包装 `ray job submit`
  - 捕获 `No available agent to submit job`
  - 在超时时间内自动重试

### 验证结果

后续训练可以正常提交 Ray job。成功运行的 job：

```text
raysubmit_tXbyaRpDihgycHS8
status=RUNNING
```

## 问题 3：训练与 rollout colocate 后显存压力过大

### 表现

Qwen3.5-9B colocate 训练和 SGLang rollout 时，显存峰值很高。早期配置下容易在启动、权重同步、rollout 生成阶段不稳定。

另外，rollout 默认并发较高，和训练 actor 共用 8 张卡时会放大峰值。

### 原因判断

这是 colocate 训练 + rollout 的资源竞争问题。模型训练、SGLang KV cache、权重同步、optimizer state 和 checkpoint 保存都会占用显存或 host memory。

### 解决方式

曾在 `scripts/run-qwen3.5-9B.sh` 里临时降低 rollout 和 batch 压力，用于排查 colocate 显存峰值。该组问题 3 的降载改动现已按要求恢复为原脚本默认值，避免把 debug 配置作为默认训练配置。

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `--rollout-batch-size` | `32` | 恢复原脚本每轮采样 prompt 数。 |
| `--n-samples-per-prompt` | `8` | 保持原值。 |
| `--global-batch-size` | `256` | 恢复原脚本全局 batch。 |
| `--sglang-mem-fraction-static` | `0.7` | 恢复原脚本 SGLang 静态显存比例。 |
| `--sglang-max-running-requests` | 未显式设置 | 不再在脚本默认限制 SGLang running requests。 |
| `--sglang-server-concurrency` | 未显式设置 | 不再在脚本默认限制 SGLang server 并发。 |
| `--no-offload-train` | 未传入 | 不再强制关闭 train offload。 |

当前脚本保留的是原始 colocate 参数：

```bash
--colocate
--rollout-batch-size 32
--n-samples-per-prompt 8
--global-batch-size 256
--sglang-mem-fraction-static 0.7
```

如果后续还需要定位 colocate 显存问题，建议另做 debug preset 或参数 sweep，不再直接改默认训练脚本。

### 验证结果

问题 3 的降载脚本改动已回退，当前未重新用原始 rollout/SGLang 参数做长跑验证。已验证过的 checkpoint 保存和 optimizer 保存修复见问题 6。

## 问题 4：pinned CPU tensor backup 在当前环境下不稳定

### 表现

训练过程中需要把 tensor 备份到 CPU。原逻辑固定使用 pinned CPU memory：

```python
torch.empty_like(param, device=torch.device("cpu"), pin_memory=True)
```

这类路径在当前 CUDA/驱动/容器组合下容易触发 `AcceleratorError` 或 allocation 相关异常。

### 原因判断

pinned host memory 不是训练正确性的必要条件，只是性能优化。当前环境里 pinned allocation 不稳定时，应自动降级到普通 CPU tensor。

### 解决方式

修改 `slime/utils/tensor_backper.py`：

- 增加环境变量：

```bash
SLIME_TENSOR_BACKUP_PIN_MEMORY=0
```

- 当 pinned allocation 失败时自动 fallback 到非 pinned CPU tensor。
- `scripts/run-qwen3.5-9B.sh` 的 Ray runtime env 中固定传：

```json
"SLIME_TENSOR_BACKUP_PIN_MEMORY": "0"
```

### 验证结果

后续训练可以完成权重备份、权重同步、rollout 和 train，不再卡在 pinned CPU tensor backup。

## 问题 5：初始 eval 和 eval 数据路径导致启动阶段额外不确定性

### 表现

原始 eval 配置使用过不稳定/不一致路径，例如 `/root/aime-2024/aime-2024.jsonl`。同时启动前 eval 会让 debug 周期变长，并引入额外显存和 SGLang 负载。

### 原因判断

当前目标是先把 Qwen3.5-9B RL 训练主链路跑通，启动前 eval 不是必要路径。

### 解决方式

修改 eval 参数：

```bash
--skip-eval-before-train
--eval-prompt-data aime ./data/aime-2024/aime-2024.jsonl
--n-samples-per-eval-prompt 1
```

### 验证结果

训练启动后直接进入 rollout -> train -> save 主链路，减少了调试噪音。

## 问题 6：optimizer checkpoint 保存崩溃

### 表现

训练先前可以跑到较后步骤，但保存 checkpoint 时崩溃。关键报错：

```text
saving checkpoint at iteration 19
Storing distributed optimizer sharded state of type dp_reshardable
...
/root/Megatron-LM/megatron/core/dist_checkpointing/strategies/filesystem_async.py line 226
tensor.to("cpu", non_blocking=non_blocking)
torch.AcceleratorError: CUDA error: invalid argument
```

当时半写入的 checkpoint 已移到：

```text
checkpoints/Qwen3.5-9B/iter_0000019.failed_20260512_0410
```

### 原因判断

崩溃发生在 Megatron Core distributed optimizer checkpoint 的默认 `dp_reshardable` 格式保存路径中。该格式在当前环境下会走到 `filesystem_async.py` 的 GPU -> CPU non-blocking copy，并触发 `CUDA error: invalid argument`。

临时绕过曾经加过 `--no-save-optim`，但这会导致 optimizer state 不保存，不能作为最终方案。

### 解决方式

最终修复在 `scripts/run-qwen3.5-9B.sh`：

- 删除 `NO_SAVE_OPTIM` 逻辑。
- 不再传 `--no-save-optim`。
- 启用 Megatron Core 官方支持的 fully reshardable optimizer checkpoint 格式：

```bash
--dist-ckpt-optim-fully-reshardable
--distrib-optim-fully-reshardable-mem-efficient
```

为了快速验证，同时把：

```bash
SAVE_INTERVAL=${SAVE_INTERVAL:-1}
```

### 验证结果

日志确认 optimizer 没有被跳过：

```text
no_save_optim = False
save_interval = 1
dist_ckpt_optim_fully_reshardable = True
distrib_optim_fully_reshardable_mem_efficient = True
```

日志确认保存的是 optimizer fully reshardable 格式：

```text
saving checkpoint at iteration       0 to checkpoints/Qwen3.5-9B/ in torch_dist format
Storing distributed optimizer sharded state of type fully_reshardable
Timer save_model end (elapsed: 359.9s)

saving checkpoint at iteration       1 to checkpoints/Qwen3.5-9B/ in torch_dist format
Storing distributed optimizer sharded state of type fully_reshardable
Timer save_model end (elapsed: 312.5s)

saving checkpoint at iteration       2 to checkpoints/Qwen3.5-9B/ in torch_dist format
Storing distributed optimizer sharded state of type fully_reshardable
Timer save_model end (elapsed: 319.3s)
```

成功生成：

```text
latest_checkpointed_iteration.txt = 2
iter_0000000 117G
iter_0000001 117G
iter_0000002 117G
```

没有再出现 `filesystem_async.py` 的 CUDA invalid argument。

## 当前整理后的提交结构

为了这次 review，`f27762a4 Update run-qwen3-9B script to correct eval prompt data path` 之后的改动已经重新整理：

```text
f27762a4 基线 commit
a8c10805 Merge latest main into dev_csl
<本 handoff 所在 proposed commit> Qwen3.5-9B 训练修复与验证记录
```

旧的细分提交保存在本地备份分支：

```text
backup/dev_csl-before-review-rewrite-20260512_102058
```

旧历史里曾短暂尝试过 `--no-save-optim`，但这个方向已经废弃。本次 proposed commit 不包含该绕过，最终方案会保存 optimizer checkpoint。

## 与最新 main 的关系

最新 main 合入了 `Patch Megatron TP grad coalesce to chunked all-reduce (#1899)`。它能降低 tensor parallel grad sync 的峰值连续显存申请，和本文“问题 3：训练与 rollout colocate 后显存压力过大”有关。

它没有覆盖以下问题，所以这些仍保留在 proposed commit 中：

- Ray dashboard agent 端口文件超时和 job agent ready 竞态。
- pinned CPU tensor backup fallback。
- eval 路径与启动前 eval 的调试噪音。
- optimizer checkpoint 使用 fully reshardable 格式保存。

## 如何恢复训练

如果接受当前 `SAVE_INTERVAL=1`，可以直接启动原容器：

```bash
ssh chenshuailin@192.168.16.55 'docker start slime-qwen9b-train-current'
```

它会从最新 checkpoint marker 恢复，即 iteration `2`。

如果是长跑，建议先把 `scripts/run-qwen3.5-9B.sh` 里的 debug 保存间隔调大，例如：

```bash
SAVE_INTERVAL=${SAVE_INTERVAL:-20}
```

再重启容器。

如果重建容器，需要保留：

```bash
MASTER_ADDR=192.168.16.55
NODE_ADDR=192.168.16.55
```

## 注意事项

- 当前 `SAVE_INTERVAL=1` 是为了快速 debug checkpoint；长跑会每步写约 `117G`。
- 现在已有 3 个有效 checkpoint，约占 `351G`。
- `/nfs/FM` 当前可用空间约 `17T`，但继续每步保存会很快增长。
- `scripts/ray/start_cluster.sh` 会杀掉容器内所有 Python 进程，因此只应在专用训练容器里使用。
- 如果不再需要旧失败目录，可以删除：

```bash
rm -rf checkpoints/Qwen3.5-9B/iter_0000019.failed_20260512_0410
```
