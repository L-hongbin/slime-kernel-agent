# DrKernel 评测教程

本教程介绍如何对模型在 KernelBench 数据集上进行评测（eval-only 模式，不训练）。



### 1.1 Docker 环境搭建与初始化

推荐使用官方提供的开发 Docker 镜像。请执行以下命令创建并启动容器：

```bash
docker run -itd --gpus all \
  --device /dev/infiniband \
  -v /nfs:/nfs \
  --ipc host \
  -p 23422:22 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --privileged \
  --name csl_slime \
  192.168.14.129:80/library/slime:nightly-dev-20260530a bash
```

进入容器后，需要在仓库根目录下执行环境初始化脚本，以配置软链接并以可编辑模式安装包：

```bash
# 进入容器并切换到仓库目录
docker exec -it csl_slime bash
cd /path/to/slime   # 切换到你的 slime 仓库根目录

# 运行环境初始化脚本
bash ./set_env.sh
```

## 2. 评测流程概览

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  启动 Ray 集群   │───▶│  启动 SGLang 引擎 │───▶│  加载模型权重    │
└─────────────────┘    └──────────────────┘    └────────┬────────┘
                                                        │
                            ┌───────────────────────────▼──────────────┐
                            │           评测循环（每个 prompt）          │
                            │                                          │
                            │  Turn 1: 渲染 prompt → SGLang 生成       │
                            │       → KernelGym 编译+benchmark         │
                            │       → 获取 reward（speedup）            │
                            │                                          │
                            │  Turn 2: 拼接 feedback → SGLang 生成     │
                            │       → KernelGym 编译+benchmark         │
                            │       → 获取 reward                      │
                            │                                          │
                            │  Turn 3: 同上                            │
                            └───────────────────────────┬──────────────┘
                                                        │
                                                        ▼
                            ┌──────────────────────────────────────────┐
                            │  汇总指标 & 保存详细数据（eval_0.pt）      │
                            └──────────────────────────────────────────┘
```


## 3. 快速开始

### 3.1 使用示例脚本

仓库提供了一个注释完善的示例脚本：

```bash
# 直接运行（使用默认参数）
bash scripts/eval_drkernel_example.sh

# 自定义参数
CTX_LEN=32768 \
N_SAMPLES=4 \
TP=4 \
MODEL_DIR=/path/to/your/model \
RM_URL=http://your-kernelgym-server:20111 \
bash scripts/eval_drkernel_example.sh
```

脚本的每个关键参数都有中文注释，建议打开 [`scripts/eval_drkernel_example.sh`](../../../scripts/eval_drkernel_example.sh) 阅读。

### 3.2 常用环境变量

| 变量                     | 默认值                        | 说明                                          |
| ------------------------ | ----------------------------- | --------------------------------------------- |
| `MODEL_DIR`              | `.../Qwen3.6-27B`             | HuggingFace 模型路径                          |
| `CTX_LEN`                | `65536`                       | 上下文窗口（prompt + response 总 token 上限） |
| `N_SAMPLES`              | `8`                           | 每个评测 prompt 的采样数（越大越稳定）        |
| `TP`                     | `2`                           | Tensor Parallel 大小                          |
| `RM_URL`                 | `http://192.168.16.39:20111`  | KernelGym Reward Server 地址                  |
| `DRKERNEL_GPU_NAME`      | `NVIDIA GeForce RTX 4090 ...` | 注入 prompt 的目标 GPU 信息                   |
| `DRKERNEL_COMPILER_NAME` | `CUDA 12.9 ...`               | 注入 prompt 的编译器信息                      |

---

## 4. 核心概念

### 4.1 Eval-Only 模式的两个关键参数

```bash
--num-rollout 0          # 不做训练 rollout，直接触发 eval
--debug-rollout-only     # 跳过训练步骤（仅 rollout + eval）
```

`train.py` 中的逻辑（[train.py:40-41](../../../train.py#L40-L41)）：

```python
# special case for eval-only
if args.num_rollout == 0 and args.eval_interval is not None:
    ray.get(rollout_manager.eval.remote(rollout_id=0))
```

### 4.2 评测配置文件（eval config YAML）

评测数据集通过 `--eval-config` 指定一个 YAML 文件，格式如下（[eval_kernelbench_level1.yaml](../../../scripts/eval_kernelbench_level1.yaml)）：

```yaml
eval:
  defaults:
    temperature: 1.0
    top_p: 0.95
    top_k: 20
  datasets:
    - name: kernelbench_level1
      path: data/kernelbench-level1-validation/train.parquet
      input_key: ground_truth        # parquet 中存放 kernel 源码的字段
      label_key: ground_truth
      metadata_key: extra_info       # 元数据字段（含 problem_id 等）
      metadata_overrides:            # 注入到每个 sample 的 metadata
        benchmark: kernelbench
        entry_point: Model
        is_valid: true
        level: 1
        split: validation
```

你也可以添加多个 datasets 来同时评测不同数据集。

### 4.3 数据格式

评测数据为 Parquet 格式，每行包含：

| 字段           | 说明                         | 示例                                                                |
| -------------- | ---------------------------- | ------------------------------------------------------------------- |
| `ground_truth` | 待优化的 PyTorch kernel 源码 | `import torch\nclass Model(nn.Module):...`                          |
| `extra_info`   | 元数据字典                   | `{"problem_id": 1, "name": "1_Square_matrix_multiplication_", ...}` |

可以通过 Python 快速查看：

```python
import pandas as pd
df = pd.read_parquet("data/kernelbench-level1-validation/train.parquet")
print(df.columns.tolist())  # ['ground_truth', 'extra_info']
print(len(df))               # 100 (level-1 验证集)
```

### 4.4 `--dump-details` 输出

指定 `--dump-details ${SAVE_DIR}/dumps` 后，评测数据会保存到：

```
${SAVE_DIR}/dumps/
├── rollout_data/
│   └── eval_0.pt              # 评测详细数据（所有 sample 的完整信息）
└── train_data/                # eval-only 时为空
```

`eval_0.pt` 是一个 `torch.save` 的字典，包含所有评测样本的详细信息：

```python
import torch
data = torch.load("dumps/rollout_data/eval_0.pt", weights_only=False, map_location="cpu")

# data["samples"] 是一个 list，每个元素是一个 sample dict
sample = data["samples"][0]
print(sample.keys())
# dict_keys(['prompt', 'response', 'reward', 'metadata', ...])

# metadata 包含每轮的 KernelGym 评测结果
turns = sample["metadata"]["turns"]
for i, turn in enumerate(turns):
    kg = turn.get("kernelgym", {})
    print(f"Turn {i+1}: compiled={kg.get('compiled')}, "
          f"correctness={kg.get('correctness')}, "
          f"speedup={kg.get('speedup')}")
```

---

## 5. 分析评测结果

### 5.1 查看终端日志中的 score

评测完成后，终端日志会打印汇总指标：

```
eval 0: {'eval/kernelbench_level1': 0.38125, ...}
```

其中 `eval/kernelbench_level1` 就是最终 **score**（所有 sample 的平均 reward）。

### 5.2 分析每轮精度

使用仓库内置的分析脚本：

```bash
# 每轮的编译率、正确率、加速比
python3 scripts/analysis/per_turn_acc.py ${SAVE_DIR}/dumps/rollout_data/eval_0.pt
```

输出示例：

```
N=800  (in_all 分母=全部样本)
         T1     T2     T3
Comp  38.1   62.4   65.8
Corr  22.2   36.5   38.2
F1.0   9.0   15.4   16.0
F1.2   4.6    8.9    9.6
```

| 指标 | 含义                                         |
| ---- | -------------------------------------------- |
| Comp | 编译成功率                                   |
| Corr | 正确率（编译成功且结果正确）                 |
| F1.0 | speedup ≥ 1.0 的比例（比 baseline 快或持平） |
| F1.2 | speedup ≥ 1.2 的比例（比 baseline 快 20%+）  |

### 5.3 分析每轮 token 长度

```bash
python3 scripts/analysis/per_turn_len.py ${SAVE_DIR}/dumps/rollout_data/eval_0.pt
```

---

## 6. 常见配置变体


### 6.3 减少采样数加速调试

```bash
N_SAMPLES=1 bash scripts/eval_drkernel_example.sh
```

每个 prompt 只采样 1 次，wall 大幅缩短（约 8×），但 score 方差更大。

### 6.4 单轮评测

```bash
# 在 DRKERNEL_PLUGIN_ARGS 中把 max-turns 改为 1
# 或修改环境变量（需在脚本中支持）
```

---

## 7. 目录结构参考

评测完成后，输出目录结构如下：

```
checkpoints/Qwen3.6-27B/20260601_120000_eval_example_ctx65536_n8/
├── run.log                           # 完整运行日志
├── eval_config.resolved.yaml         # 解析后的评测配置
└── dumps/
    └── rollout_data/
        └── eval_0.pt                 # 评测详细数据
```
