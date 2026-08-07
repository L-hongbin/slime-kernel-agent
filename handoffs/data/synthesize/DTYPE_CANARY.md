# Dtype 1k canary 交接（parameter-free + module-state）

## 结论和范围

本 lane 不需要 LLM。两个保守子集均由确定性 solver 构造并由 paired reference、三轮 runtime proof 和 dispatch trace 验收：历史 parameter-free canary 得到 638 条；本轮带 parameter/buffer 的 `module_state` coherent canary 得到 479 条。LLM 最多只能提出更宽候选，不能替代静态证明或 runtime 证据。

`module_state` child 同时改写直接 FP32 floating input factory，并在 `Model.__init__` 末尾显式执行基类 `torch.nn.Module.to(self, dtype=target)`。它只接纳可静态归因到内置 `torch.nn` 注册 state 的模型；runtime 再逐项证明 parameter/buffer 是 parent FP32 state 的精确低精度 cast、非浮点 state 不变、alias 保持、无未注册 tensor/module state、构造 RNG 一致、forward 后 state 仍一致且核心 dispatch 没有 FP32/complex fallback。

solver、liveness 和 exact analyzer 均有 5,000-row 硬门禁；两个 canary 都固定为 1,000 个 parent/child。所有 materialized row 均为 `dtype_intervention_review_only`、`training_approved=false`，没有进入训练数据。

## Parameter-free solver 和语义合同（历史 canary）

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

## Module-state coherent 1k canary（本轮）

### 静态合同

`--coherence-class module_state` 复用同一套入口，不另建 v2 pipeline。solver 只接受 canonical `torch.nn.Module`、内置 `nn` state constructor、scalar-only `get_init_inputs`，并拒绝 custom base/decorator、custom `to/_apply/getattr/setattr`、model 内显式 dtype cast、无法归因的 tensor constant/state 和动态 constructor。child 的唯一 dtype intervention 是：

1. 所有直接 FP32 floating input factory 改为稳定分配的 FP16 或 BF16；
2. 在 `Model.__init__` 末尾注入 `torch.nn.Module.to(self, dtype=torch.<target>)`；
3. shape、layout、value family、forward AST 和 init inputs 保持不变。

Runtime validator 对相同 init inputs、相同构造 RNG 的 parent/child 各跑 3 trials。每轮要求 registered floating state 精确等于 parent state cast，non-floating state 精确不变，name/metadata/object alias/storage alias 保持；递归扫描并拒绝 list/dict 中隐藏的 tensor/module state，检查 non-persistent buffer 和 forward 后 state；同时拒绝 FP32/complex dispatch fallback。64 GiB guard 同时约束实际 CUDA allocator 以及 parent+child registered-state bytes。

Canonical parent 仍为 `Data/prompt_tvm_v4/train.review.parquet`，64,315 rows。静态 eligible 为 20,567，固定选择 1,000 条：BF16 497、FP16 503；source 为 cuda_agent 120、drkernel 745、kernelbook 91、oubo_generated 44；operator bucket 为 2–5 ops 792、≥6 ops 184、≤1 op 24。generator commit 为 `95c545af54a5264dc7f9c0b8a889be48757fae4e`，generator SHA 为 `a10716de031876941d73a51f19e7dc8ab3730a62971ae9b3300db1ea86ea5dfd`。

23 个静态样本覆盖四个 source、两个 dtype 和三个 operator bucket。人工 diff 确认只有 AST 等价重排/注释丢失、direct factory dtype 和 `Module.to` 注入；1,000 个 parent/child/selection UUID 均唯一。

| Static artifact | SHA-256 |
| --- | --- |
| `parents.parquet` | `4cf4c06595d4fef989635d0ed90e4212d80d507beeebcf524cc8e160ef8058e5` |
| `candidates.parquet` | `678a1edfa5f4b4004d6051836f814422172f18f4b5b548cc4c6b364dae0d6ee3` |
| `paired.parquet` | `095607f8a4b52ac8476114482cda0efdd5d456d5cb3e970e0028e1e0a3a0fc29` |
| `manifest.jsonl` | `f371dcda21b88d2257ef09dc1b915e91ab20f5c4d70dcd4137bc4cf72e7c151e` |
| `decisions.jsonl` | `b6d6956b26655e644803e8c5e038a6c47f87e0b0b77e27b8c3768ab92fac0a9a` |
| `review_samples.md` | `799de340f4377770e31a617227387619759d0050c297966e512fd87d9fa9c854` |

### A800 reference 和 liveness

运行环境为 node22 的 8× NVIDIA A800-SXM4-80GB、Torch `2.11.0+cu129`、CUDA 12.9、cuDNN 91701、KernelGym commit `26255057463a77b23abac0f3e5eafeeebf2ebbb5`。LSTM 使用 `RUNTIME.md` 记录的 cuDNN 9.17.1 library path；node21 因 CUDA context 异常未参与本轮产出，避免把节点故障混入方法通过率。

Paired reference 使用 persistent train mode、5 trials、180 秒逐行 timeout 和 64 GiB guard。2,000 个 parent/child rows 的结果为：

| Reference class | BF16 | FP16 | Total |
| --- | ---: | ---: | ---: |
| both pass | 425 | 438 | 863 |
| both fail | 61 | 58 | 119 |
| parent pass / child fail | 11 | 6 | 17 |
| parent fail / child pass | 0 | 1 | 1 |

只有 863 个 both-pass child 进入 liveness，allowlist SHA 为 `e30237eac7ea24c07382a79d9eda150be649f8995f065590d778925260194877`。其中 479 pass、380 unsupported、4 OOM：

| Target | Passed | Unsupported | OOM |
| --- | ---: | ---: | ---: |
| BF16 | 218 | 204 | 3 |
| FP16 | 261 | 176 | 1 |
| **Total** | **479** | **380** | **4** |

384 个拒绝按稳定原因前缀归并为：FP32/complex dispatch fallback 178、cast-equivalent output mismatch 91、parent exact control failure 63、runtime unregistered tensor state 31、non-float output mismatch 10、registered floating state missing 5、OOM 4、post-`get_inputs` RNG change 2。通过的 BF16/FP16 分别覆盖 654/783 trials、17,616/13,836 dispatch calls，accepted 的 FP32/complex fallback 均为 0。

最终 accepted 为 BF16 218、FP16 261；source 为 cuda_agent 44、drkernel 345、kernelbook 73、oubo_generated 17；operator bucket 为 2–5 ops 386、≥6 ops 76、≤1 op 17。全量结构统计显示，11 条含 buffer、1 条含 non-floating state、2 条含需要保持的 object/storage alias group；人工抽取的 8 个 source×dtype cell 均有 3 份一致的 state evidence。代表例包括：

- `dtype_e719c10f9a1c2c5151914d75`：2 parameters + 1 buffer，state bytes 640→320；
- `dtype_2e7c35e7d8cb59f089247a6e`：4 parameters，其中 1 个 non-floating state 精确保持；
- `dtype_607ce8aa826e38e0fd6e0f78`：8 parameters，4 个 object alias 与 4 个 storage alias 均保持，state bytes 3584→1792。

### Materialization 和审查

不可变 canary 目录为：

`local_artifacts/data_handoffs/prompt_tvm_v4_dtype_module_state_canary1000`

| Artifact | Rows / files | SHA-256 |
| --- | ---: | --- |
| `runtime/accepted.parquet` | 479 | `76dbcc21e643d3a2fec6e9962a242c5b852a36d61358edf4c02669c7cb32d46a` |
| `runtime/accepted.manifest.jsonl` | 479 | `f496078652fa0df83e8f1c9fdc46dc6467728f9df85dfea3caa014d9db74d974` |
| `runtime/raw_artifact_sha256.json` | raw evidence | `99df85cbd058c675848b04117dc669d6f67e1af8b08235581b7426ba1e2f5e99` |
| `runtime/final_summary.json` | — | `261a935db58ec72a39cd33d09b401c0af0f2eb6faeb524ceafa8206bac97dccc` |
| `analysis/reference_summary.json` | — | `4f6f15699b7a834946f1cda70dbd6b36fe83c57b629cb76d561a086fd7f91048` |
| `analysis/liveness_summary.json` | — | `7a245908214b07e0fb1526985fce2810f99e0a558302a7b599b0c5102cfd0db6` |
| `analysis/failure_bias_report.md` | — | `38f31b5317ea513f36da0bc51a36145bdb7e4dce89d439cb36c0db8a5d58a4e0` |

Analyzer SHA 为 `5db4b4566805b306a7d004f73b9676e53eddbda9c4dfea714938870979ad3c3d`，runtime policy fingerprint 为 `88575e6625b057d0973cfc7d46982dcf6d1b0a1047a6b9cd4419d07eae020632`。479 条 manifest 均记录 `rng_consumption_status=construction_rng_equal_observed`、`materialization_status=dtype_intervention_review_only`、`training_approved=false`。

Kimip 在实现、静态 1k 和最终证据三个 milestone 做了只读审查。实现阶段发现 list/dict 内隐藏 unregistered module 的 fail-open 路径，已修复；静态审查无 P0/P1；最终审查 verdict 为 PASS、无 P0/P1，并发现 accepted manifest RNG 标签仍为 `runtime_pending` 的 P2。该 P2 已在 analyzer 合并处修复并用 Python 3.12 exact replay 重新 materialize，accepted 集合仍为 479；修复后复核为 P0/P1/P2 均无、verdict PASS。

### Module-state 复现入口

Python 3.12 是 source-bound exact AST replay 环境；Python 3.13 的 `ast.unparse` 文本差异会被 fail-closed 拒绝。

```bash
lane_dir=local_artifacts/data_handoffs/prompt_tvm_v4_dtype_module_state_canary1000
PYTHONPATH=. uv run --python 3.12 --no-project --with 'pyarrow==24.0.0' \
  python tools/data/synthesize/dtype_method/solve_dtype_coverage.py \
  Data/prompt_tvm_v4/train.review.parquet "${lane_dir}" \
  --limit 1000 --coherence-class module_state
```

Reference 双通过 allowlist 生成后：

```bash
bash tools/data/synthesize/dtype_method/launch_dtype_liveness_shards.sh \
  "${lane_dir}/parents.parquet" \
  "${lane_dir}/candidates.parquet" \
  "${lane_dir}/manifest.jsonl" \
  "${lane_dir}/analysis/reference_both_pass_child_uuids.txt" \
  "${lane_dir}/runtime/liveness/node22"

PYTHONPATH=. uv run --python 3.12 --no-project --with 'pyarrow==24.0.0' \
  python tools/data/synthesize/dtype_method/analyze_dtype_run.py \
  "${lane_dir}" \
  --reference-dir "${lane_dir}/runtime/reference/node22" \
  --liveness-dir "${lane_dir}/runtime/liveness/node22"
```

代码验证包括 Black、Ruff、`py_compile`、shell syntax、static exact replay、隐藏 state negative smoke、A800 registered-state/alias targeted smoke、2,000-row paired reference 和 863-row liveness。按生产数据 pipeline 约定，本轮没有新增单元测试。

## 尚未证明的内容

这两份 canary 证明的是两个保守子集在冻结 A800 环境和有限 trials 下的低精度可执行性与状态合同，不证明所有 seed、硬件、operator 或任意自定义 module 都语义等价。`module_state` 也没有覆盖 dtype-sensitive 常量、自定义 conversion hook、动态/未注册 state；这些 case 应继续 fail-closed，不能用 LLM 文本判断替代证据。638+479 条 accepted 仍需 use-site review、cross-lane evidence 统一和受控训练 ablation；在这些步骤完成前不得设置 `training_approved=true`。
