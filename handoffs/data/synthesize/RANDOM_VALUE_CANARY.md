# Random/value 1k canary 交接

## 结论和边界

本轮不需要 LLM。四种 random/value 方法都可以由确定性 solver 构造：`uniform_01`、`signed_uniform`、`poisson_counts`、`multinomial_categories`。正式实现不生成 `boundary_pm1`，也不把 Poisson 或 multinomial 解释成通用连续输入替换；它们只是保持原 shape/dtype/layout 的受约束 value coverage cell。

本机只负责小批量验证。solver CLI 必须显式传 `--limit`，并在代码中限制为 1–5,000；liveness 和 exact analyzer 另有独立的 5,000-row 硬门禁。本轮固定为 1,000 个 parent/child，不自动扩到 44,380-row eligible pool。所有产物保持 `training_approved=false`。

2026-08-05 的代码复核后，random/value 已收敛为四个无版本后缀的正式入口。复核同时加固了 canonical row 锚定、reference fingerprint 重算、至少三次 liveness、单一 runtime partition、动态 dtype 的 Multinomial 精确表示检查和 pinned-memory fail-closed。下文的 1k 数字来自加固前的 source-bound canary，只作为不可变历史观测；由于 generator/analyzer source SHA 已变化，当前代码会拒绝把旧 manifest 重新 materialize。新的训练候选必须用当前正式入口重建，不能给旧 evidence 换名或放松 source hash。

## 四种构造

- `uniform_01`：原 `randn` draw 只求值一次，用 Normal CDF 映射到 `[0, 1)`；
- `signed_uniform`：原 draw 只求值一次，用 `erf` 映射到 `[-1, 1)`；
- `poisson_counts`：由原 draw 计算 `softplus` rate，再 clamp 到 `[0.125, 8]`，通过 per-factory stable seed 的局部 generator 调用 `torch.poisson`；输出仍为原浮点 dtype，但取非负整数值；
- `multinomial_categories`：仅接受静态 last dimension ≥2 且 category ID 可由原 dtype 精确表示的 factory；沿最后一维 softmax，带放回采样 `last_dim` 个 category，再恢复原 shape 并转回原 dtype。

Poisson 和 multinomial 的局部 generator 绑定原 draw 的 device，不推进默认全局 generator。静态合同逐行证明并重放：`Model`、`get_init_inputs`、shape、dtype、layout 不变，原全局 `randn` 消耗不变；runtime 继续核对 device、stride、storage offset、requires-grad、post-`get_inputs` RNG、目标 support、unchanged-parent control 和输出效应。

## 静态 1k 构造

Canonical parent：

- rows：64,315；
- SHA-256：`b07205fcadc543964cfc7ee5fd9c1e4d011f0f3481447f656e5297e40b4b99f4`。

Eligible pool 为 44,380；只报告可扩展性，不构成放量授权。1k 分布：

| Family | Candidates |
| --- | ---: |
| `multinomial_categories` | 246 |
| `poisson_counts` | 254 |
| `signed_uniform` | 253 |
| `uniform_01` | 247 |

Source 为 cuda_agent 109、drkernel 845、oubo_generated 46；operator bucket 为 2–5 ops 833、≥6 ops 130、≤1 op 37。

静态 artifact SHA：

| Artifact | SHA-256 |
| --- | --- |
| `parents.parquet` | `c30a3d037e780cba970e69ee1097b1eb60152cc8af35b14cbf94aa3aed15e2ab` |
| `candidates.parquet` | `a913242fe0fa185b0d3f89783ff63feaf5da44fbe31ef65799658e2674e705dd` |
| `paired.parquet` | `9614f25e9ead88c8160cdf4cb2e8a19e88ecc90abdef28e2238de2d225d09c81` |
| `manifest.jsonl` | `cd6816b0c1bd9d599d9c37ae6f3affba768018840b953eca30de5ec2fc7c4bf4` |
| `decisions.jsonl` | `f6625e32f52070db5b4e7270ef72a3343345751b1fd1c4565927dae509661f31` |
| `review_samples.md` | `cf82b8a44babaa42382f6d6524a18f84cab2c9fabf8316f735bfbee369814bc8` |

Parquet 构造和最终 materialization 固定使用 `pyarrow==24.0.0`。

## A800 paired reference

这份历史 canary 在 node22 的 8× NVIDIA A800-SXM4-80GB 上运行：Torch `2.11.0+cu129`、CUDA 12.9、cuDNN 91701、KernelGym commit `26255057463a77b23abac0f3e5eafeeebf2ebbb5`。Reference 使用 persistent train mode、paired RNG reset、5 trials、180 秒逐行 timeout 和 64 GiB allocator guard；contract fingerprint 为 `542bf69d1e71e147bff65ff17e4a96595c3d3bb0c9b597fef6181cb90ca64ae7`。

| Family | Both pass | Both fail | Parent pass / child fail |
| --- | ---: | ---: | ---: |
| `multinomial_categories` | 213 | 28 | 5 |
| `poisson_counts` | 220 | 32 | 2 |
| `signed_uniform` | 211 | 42 | 0 |
| `uniform_01` | 206 | 40 | 1 |
| **Total** | **850** | **142** | **8** |

142 个双失败归入 baseline/evaluator。8 个 child-only failure 中：Multinomial 为 3 correctness、1 OOM、1 环境瞬态；Poisson 为 1 correctness、1 timeout；Uniform `[0,1)` 为 1 correctness；signed Uniform 为 0。只有 850 个双通过 child 进入 liveness allowlist，allowlist SHA 为 `59d47b9f3760504f161026d115f751720989f121c1dd0d97f34260ed2c754dce`。

## A800 value/output liveness

850 个 reference 双通过 child 各跑 3 trials，结果为 792 pass、58 reject：

| Family | Passed | Output-effect rejected | Parent-control unsupported | Runtime failure | OOM |
| --- | ---: | ---: | ---: | ---: | ---: |
| `multinomial_categories` | 205 | 0 | 8 | 0 | 0 |
| `poisson_counts` | 209 | 1 | 8 | 1 | 1 |
| `signed_uniform` | 189 | 8 | 14 | 0 | 0 |
| `uniform_01` | 189 | 5 | 12 | 0 | 0 |
| **Total** | **792** | **14** | **42** | **1** | **1** |

唯一 runtime failure 是一次 CUDA driver initialization 瞬态，仍按 fail-closed 拒绝；OOM 是一个 Poisson child 在 64 GiB policy 下额外申请 8.25 GiB 时被拒绝。没有 shape、dtype、layout、global-RNG 或 family-support 违规。

Passed Poisson 覆盖 209 个 child、627 个 raw trials、735 个 tensor observations，实际观测值域为 0–17。Passed Multinomial 覆盖 205 个 child、615 个 raw trials、696 个 tensor observations，last-dimension cardinality 从 2 到 200,000；完整 cardinality 计数见 `analysis/liveness_summary.json` 和 `analysis/failure_bias_report.md`。

最终 accepted 分布：

- family：Multinomial 205、Poisson 209、signed Uniform 189、Uniform `[0,1)` 189；
- source：cuda_agent 72、drkernel 677、oubo_generated 43；
- operator bucket：2–5 ops 670、≥6 ops 91、≤1 op 31。

## 历史 1k runtime 观测

加固前 canary 的原始目录保留为不可变审计材料：

`local_artifacts/data_handoffs/prompt_tvm_v4_random_value_v2_canary1000`

| Artifact | SHA-256 |
| --- | --- |
| `runtime/accepted.parquet` | `dc0b1e03f88513dcc6b213ab22a2f0241f9b0bbe3e02c9c6bdd48e0612b8a9a8` |
| `runtime/accepted.manifest.jsonl` | `9df4e434ff578d838475508e27b955ad7abc24f8055c69dc98205aafecf69d44` |
| `runtime/final_summary.json` | `47aad0a9e7461d935c8d1795e941dacf97e6035e84dd2cd0da102f725ce20c46` |
| `runtime/raw_artifact_sha256.json` | `a217d030c4c5e9c9e229e30c3cc37acad885add8a06c37d95eadc2e354fbf158` |
| `analysis/reference_summary.json` | `8475f1f487422b9b8c70c37e2b9dd43233177d20548cb44771ce095c27b9c7c9` |
| `analysis/liveness_summary.json` | `4f61841a359a9ff19db7374fbc519ac5c7e0536276182b34c743f42aa2b959da` |
| `analysis/failure_bias_report.md` | `14581ce4075740f435b4da624c16cf106c45c2b424b2b9c6ff0eb67c9b79afeb` |

Raw manifest 精确覆盖 reference/liveness 两个权威目录的 38 个非隐藏文件。Runtime policy fingerprint 为 `e3f5bcc8a1c716778b5b33cd8b2f1eb419df4634bc8cc2ae47a435148344dcb8`；liveness binding 为 `43554f85344ade897d35044aaa21722a04933fa26280fb1e11caabf837a57ea4`。

独立复核确认该目录的 accepted parquet/manifest UUID 顺序一致、四方法数量一致、全部 `training_approved=false`，且 raw manifest 的 38 个 SHA 全部匹配。但是，它由加固前的 source-bound 工具生成：旧 accepted parquet 的内嵌 runtime 状态仍为 pending，Poisson evidence 只有 min/max 而没有 rate/count histogram。目录名及文件不能改名、覆盖或用当前 analyzer 重新签发；上面的通过数只作为历史观测，不是当前代码下可直接发布的 accepted partition。

## 当前正式入口

random/value 只有以下四个无版本后缀入口：

- `tools/data/synthesize/random_method/solve_value_coverage.py`
- `tools/data/synthesize/random_method/validate_value_liveness.py`
- `tools/data/synthesize/random_method/launch_value_liveness_shards.sh`
- `tools/data/synthesize/random_method/analyze_value_run.py`

内部 contract version 字符串继续作为 evidence schema 身份使用；它们不是第二套源码路径。当前 analyzer 会重新打开 immutable canonical parquet、校验真实 SHA/row count/index，并在比较前按 lane schema 补入 nullable augmentation 字段；随后用当前 commit 和 solver 逐行重建 child/manifest。Reference contract payload/fingerprint、launcher/validator SHA、冻结的 KernelGym commit/bundle SHA、至少 3 次 liveness、reference/liveness 一致的单一 GPU/runtime partition 以及所有 artifact binding 都会重算。任一 source-bound byte 改变都会拒绝旧 evidence。

Poisson trial 现在从 parent 重算 solver 的 rate，并记录固定 `[0.125, 8]` 六区间 histogram、min/max/mean；count histogram 最多保留 64 个显式 bin，并对截断频次和值域做守恒校验。Multinomial 用实际 runtime dtype 的连续整数精确表示上限验收。Accepted parquet 会同步写入 runtime/governance 状态，并在 schema metadata 中显式保留 `training_approved=false`。

## 复现入口

静态构造：

```bash
lane_dir=local_artifacts/data_handoffs/prompt_tvm_v4_random_value_canary1000_hardened
PYTHONPATH=. uv run --no-project --with 'pyarrow==24.0.0' \
  python tools/data/synthesize/random_method/solve_value_coverage.py \
  Data/prompt_tvm_v4/train.review.parquet \
  "${lane_dir}" \
  --limit 1000
```

Reference 使用 `tools/data/synthesize/launch_reference_validation_shards.sh`；liveness 使用 `tools/data/synthesize/random_method/launch_value_liveness_shards.sh`。最终 exact merge：

```bash
lane_dir=local_artifacts/data_handoffs/prompt_tvm_v4_random_value_canary1000_hardened
PYTHONPATH=. uv run --no-project --with 'pyarrow==24.0.0' \
  python tools/data/synthesize/random_method/analyze_value_run.py \
  "${lane_dir}" \
  --reference-dir "${lane_dir}/runtime/reference/node22" \
  --liveness-dir "${lane_dir}/runtime/liveness/node22"
```

当前代码验证包括 Black、Ruff、`py_compile`、shell syntax、CLI/import smoke、A800 上四 family/dtype/pinned-memory/RNG-isolation smoke，不为这套生产数据 pipeline 新增单元测试。由于 source SHA 已变化，历史 1k canary 不能替代一次使用当前入口重建的小批量 GPU run。

## 语义边界

当前门禁证明的是：固定 runtime policy 下输入结构/storage 不变、全局 RNG 隔离、数值有限、family support 合规、parent control 可用，而且 child value 在每次 trial 中影响输出。它不证明 value family 与原任务具有 use-site 语义等价，也不保证任意 seed 都有效；例如把参与矩阵乘法的 Gaussian 输入变成 category ID，仍可能通过有限次 liveness。因此所有结果保持 review-only，后续仍需人工 use-site 审查、失败偏差分析和受控训练 ablation，不能把 runtime pass 改写成“语义安全”。
