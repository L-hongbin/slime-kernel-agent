# Dtype 1k canary 交接

## 结论和范围

本轮不需要 LLM。parameter-free dtype sibling 可以由确定性 solver 构造：只选择静态证明无 parameter、buffer、submodule、tensor state、tensor constant 和 model 内 dtype cast 的任务，把所有直接 FP32 floating input factory 统一改为一个稳定分配的 `float16` 或 `bfloat16` target。`Model` 和 `get_init_inputs` 的其他 AST、shape、layout、value family 及 `get_init_inputs` 全局 RNG 消耗保持不变。

带 parameter/buffer 的 coherent precision sibling 仍未实现。它原则上也可由 typed AST rewrite + runtime proof 完成，不必默认交给 LLM；但必须同时处理 parameter、buffer、dtype-sensitive constant、explicit cast 和 promotion。LLM 最多用于提出候选 rewrite，不能替代静态证明、paired reference 或 dispatch trace。本 canary 对这类任务 fail-closed，不把 input-only dtype mutation 冒充 coherent precision。

本机只做小批量验证。solver、liveness 和 exact analyzer 都有 5,000-row 硬门禁；本轮固定为 1,000 个 parent/child，不根据 11,206-row eligible pool 自动放量。最终 638 条 materialized row 全部为 `dtype_intervention_review_only`，且 `training_approved=false`。

## Solver 和语义合同

正式入口单独位于 `tools/data/synthesize/dtype_method/`：

- `solve_dtype_coverage.py`：静态 eligibility、稳定 target 分配、比例选样、child/manifest 构造和 exact replay；
- `validate_dtype_liveness.py`：三次 deterministic trial、64 GiB guard、runtime parameter-free 证明、输入 realization、cast-equivalent output 和 dispatch trace；
- `launch_dtype_liveness_shards.sh`：8 GPU source-bound shard launcher；
- `analyze_dtype_run.py`：重开 canonical source、重建每个 child、合并 paired reference/liveness 并 materialize review-only partition。

每个通过的 trial 包含两个不同目的的 child arm：

1. semantic arm 把 parent 的 FP32 输入显式 cast 到 target，运行未改变的 child `Model`，并与 parent FP32 输出按 FP16 `rtol=atol=1e-2`、BF16 `rtol=atol=2e-2` 成对比较；
2. realization arm 运行 child 自己的低精度 factory 输入，确认所有目标输入确实为 assigned dtype、目标 dtype 被实际算子消费，并拒绝任何 floating FP32 或 complex dispatch output。

这一区分是必要的：CUDA 上 dtype-specific `torch.rand` 即使保持相同 RNG 消耗，也不保证逐元素等于 FP32 draw 再 cast。本轮 passed input observations 中，FP16 为 1,100/1,239、BF16 为 1,035/1,194 与 parent cast 精确相等；不相等的 observations 来自 `torch.rand`，而 `torch.randn` observations 均相等。因此 manifest 的 `value_changed=false` 只表示没有设计新的 value-family intervention，不表示实际随机样本逐元素不变。门禁证明的是目标 dtype 下的 task/domain 可执行性、cast-equivalent 数值行为和实际低精度路径，不是形式化的全输入语义等价。

另外，三次 trial 的 direct-input schema 必须相同；输出则逐 trial 与对应 parent 比较，不要求跨 seed 的 shape 固定。`masked_select` 等数据相关输出会随随机输入改变长度，跨 seed 冻结 output shape 是错误约束。

## 静态 1k 构造

Canonical parent 为 `Data/prompt_tvm_v4/train.review.parquet`，64,315 rows，SHA-256 `b07205fcadc543964cfc7ee5fd9c1e4d011f0f3481447f656e5297e40b4b99f4`。solver source-bound commit 为 `48928f8924594f7e630b6cc7d1f158f0518ef96a`。

Eligible pool 为 11,206；target 分布为 BF16 5,644、FP16 5,562。实际 1k candidate 为 BF16 501、FP16 499；source 为 cuda_agent 72、drkernel 674、kernelbook 248、oubo_generated 6；operator bucket 为 2–5 ops 796、≥6 ops 117、≤1 op 87。

| Artifact | SHA-256 |
| --- | --- |
| `parents.parquet` | `9471d8167667a0f568ffa84ec79ccd62eddc707ba42d4ce98beaa89cd494947c` |
| `candidates.parquet` | `cb2dff7ea18a6ef4f6aa8fa0f4ab4e0bd3352c20112eef916a3fecb4187a188f` |
| `paired.parquet` | `e9b6af4a29b2fdc8e5975050a04af3331553790ee523e98eebd113a1d698645a` |
| `manifest.jsonl` | `883a64158c90799ae7626ef0a20cdfa7dd8d7916a2618da81c2622ce3b4fee11` |
| `decisions.jsonl` | `dd5baed6893b9d97fb686875c4e010bf892840da8b7de0212645a4effadece31` |
| `review_samples.md` | `59fce551c85857935dc75f4320834525f6435a9cf94a24b8813126b4810a3a3a` |

Generator SHA 为 `3b5316b2c6b7d1f70b0141d1bbced3288688b65086a440e31480a2492e980038`，共用 static dependency SHA 为 `ee5514948a46815f00378085a80935d1cf85068b78333925b35882b4e4626857`。Parquet 构造和 materialization 固定使用 `pyarrow==24.0.0`。

## A800 paired reference

canary 在 node22 的 8× NVIDIA A800-SXM4-80GB 上运行：compute capability 8.0、Torch `2.11.0+cu129`、CUDA 12.9、cuDNN 91701、KernelGym commit `26255057463a77b23abac0f3e5eafeeebf2ebbb5`。节点系统 cuDNN 9.16 与 Torch build 不匹配，因此只在 launch 环境把 `/tmp/value_lane_cudnn_9_17_1_4/nvidia/cudnn/lib` 前置到 `LD_LIBRARY_PATH`，没有修改系统 package。

Reference 使用 persistent train mode、paired RNG reset、5 trials、180 秒逐行 timeout 和 64 GiB allocator guard。2,000 个 parent/child rows 全部产生记录：

| Target | Both pass | Both fail | Parent pass / child fail |
| --- | ---: | ---: | ---: |
| BF16 | 406 | 68 | 27 |
| FP16 | 424 | 67 | 8 |
| **Total** | **830** | **135** | **35** |

135 对双失败归入 baseline/evaluator；35 个 child-only failure 中，24 个为 evaluator/environment transient，11 个为可能由 intervention 导致的 numerical/correctness failure。只有 830 个双通过 child 进入 liveness allowlist，allowlist SHA 为 `902cd26b9b62dc46e7caa71fbf5d38567e33e9312b202486d7c3b00d659a8833`。

Reference contract fingerprint 为 `5ca5269320fbdfcbf8e199f49aaf764715ab1d5132fe5cc39fdbcffaed362dea`，runtime environment fingerprint 为 `ac4ac2a6cd63618082b5a688c00a133d4a8fa8edabdf7c008865155358b98abb`。

## A800 dtype/output liveness

830 个 allowlisted child 各跑 3 trials，638 pass、192 fail-closed，无 timeout、OOM 或 worker crash：

| Target | Passed | Unsupported |
| --- | ---: | ---: |
| BF16 | 308 | 98 |
| FP16 | 330 | 94 |
| **Total** | **638** | **192** |

192 个拒绝按稳定原因前缀归并为：output tolerance mismatch 91、FP32/complex dispatch fallback 58、non-float output mismatch 29、post-`get_inputs` RNG change 7、non-finite traced output 3、output shape mismatch 2、complex input 1、runtime parameter 1。通过的 1,914 个 trials 共记录 30,024 次 semantic/realization dispatch call，FP32/complex fallback 为 0。

Liveness validator SHA 为 `4b2fb3bda96df293df66ad2721552d58afa5b5cf179d4b472c922882a1562900`，launcher SHA 为 `44afe04c0e9ac073a2f010ce6035cb449fb24edf28227c29055d6b6ea6ea00a2`，validation binding 为 `aa67cf1fabdbd178a162d52598c1ec4d92c3fb752bf5584618bb2f0f0968983f`，runtime partition fingerprint 为 `c25ab12c0ad096bbb24ede4df913dbac02b98e8c1756a5f1d301c9b9adfbdd34`。

最终 accepted 分布：

- target：BF16 308、FP16 330；
- source：cuda_agent 34、drkernel 396、kernelbook 205、oubo_generated 3；
- operator bucket：2–5 ops 488、≥6 ops 81、≤1 op 69。

## Artifact 和审计状态

不可变 canary 目录为：

`local_artifacts/data_handoffs/prompt_tvm_v4_dtype_parameter_free_canary1000`

| Artifact | Rows / files | SHA-256 |
| --- | ---: | --- |
| `runtime/accepted.parquet` | 638 | `343280a6f0e1d3ec5abc7f7808e3bc19807aca950928a6640ac77bc3d1bb0f35` |
| `runtime/accepted.manifest.jsonl` | 638 | `7e059f118f17cac4c359894797bb2bca7150e6e50d40f5b4063fe2773dad0cd6` |
| `runtime/raw_artifact_sha256.json` | 38 files | `2401a1260c2fcf1f318249d8a5ab04819b185f846625a789a05a060c875f886c` |
| `runtime/final_summary.json` | — | `8f18cdf3717e96aceec0cf5cfb11a43e9339e65331511554ee7374973dc48d7b` |
| `analysis/reference_summary.json` | — | `c6703e50988f6c578ec77d6d6632952f5d9c840ad4880a556d8161270d1e67b3` |
| `analysis/liveness_summary.json` | — | `b2ce47987e1f5c0aa09efe3096abc24d75ce45444a8393315a4c20eac7780307` |
| `analysis/failure_bias_report.md` | — | `4f1be332117a0e32491919e1afa659632231c7504de0607a5d52036b81e787fd` |

Accepted parquet metadata、638 条 accepted manifest 和 final summary 均显式记录 `training_approved=false`。Runtime policy fingerprint 为 `4c1c32220f7c59e6758928ba0644f83b4ad5dc4ca7fa4ec2d8bd348412c36d9f`。这些数据没有进入 `Data/` 或训练 union。

## 复现入口

静态构造：

```bash
lane_dir=local_artifacts/data_handoffs/prompt_tvm_v4_dtype_parameter_free_canary1000
PYTHONPATH=. uv run --no-project --with 'pyarrow==24.0.0' \
  python tools/data/synthesize/dtype_method/solve_dtype_coverage.py \
  Data/prompt_tvm_v4/train.review.parquet \
  "${lane_dir}" \
  --limit 1000
```

Reference 使用 `tools/data/synthesize/launch_reference_validation_shards.sh`。Reference 双通过 allowlist 生成后，liveness 入口为：

```bash
bash tools/data/synthesize/dtype_method/launch_dtype_liveness_shards.sh \
  "${lane_dir}/parents.parquet" \
  "${lane_dir}/candidates.parquet" \
  "${lane_dir}/manifest.jsonl" \
  "${lane_dir}/analysis/reference_both_pass_child_uuids.txt" \
  "${lane_dir}/runtime/liveness/node22"
```

最终 exact merge：

```bash
PYTHONPATH=. uv run --no-project --with 'pyarrow==24.0.0' \
  python tools/data/synthesize/dtype_method/analyze_dtype_run.py \
  "${lane_dir}" \
  --reference-dir "${lane_dir}/runtime/reference/node22" \
  --liveness-dir "${lane_dir}/runtime/liveness/node22"
```

代码验证包括 Black、Ruff、`py_compile`、shell syntax、CLI/import smoke、static exact replay、A800 targeted negative smoke、2,000-row paired reference 和 830-row liveness。按生产数据 pipeline 约定，本轮没有新增单元测试。

## 尚未证明的内容

这份 canary 证明的是一个保守 parameter-free 子集在冻结环境下的低精度可执行性和有限 trial 数值行为。它没有证明所有 seed、所有硬件或任意 operator 都保持相同语义，也没有覆盖带参数模型的 coherent conversion。638 条 accepted 仍需人工 use-site review、failure bias 审查、cross-lane evidence 统一和受控训练 ablation；在这些步骤完成前不得设置 `training_approved=true`。
