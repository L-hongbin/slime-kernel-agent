# Frontier operator scenario 1k canary 交接

## 结论（2026-08-11）

这是独立于既有 `semantic_operator_method/` 的 source-driven、parentless standalone lane，不是对原 33 个 single-Tensor templates 的放量或替代。generator 固化于 commit `e120af931b30b96fbf38eda0d6498a7925763063`；runtime output-device evidence 修复位于 `297e67d59183d8a1c98e94a60dd7db8d5067bb70`。

- deterministic constructive generator 精确生成 1,000 条、35 templates、1,000 个不含 bookkeeping `variant` 的 unique semantic-coordinate cells；
- node22 A800 KernelGym reference 为 1,000/1,000 pass；
- fresh frontier liveness v2 为 1,000/1,000 pass，最终 lane-local accepted 为 1,000/1,000；
- 所有 row 均为 `review_only`、`training_approved=false`，没有获得放量、跨 lane 合并或训练授权。

KernelGym reference 只是最低正确性基线，不是 taxonomy 的目标，也不构成数据训练价值证明。

## 构造与 coverage 合同

generator 使用固定 seed、closed registry、精确 quota、typed semantic coordinates 和 fail-closed 拒绝规则，不让 LLM 直接生成 candidate。公开论文、官方文档和公开实现只定义可审计的 operator/scenario motif；每条 manifest 都记录 `scenario_source`，不复制模型题目或 KernelBench 评测题。

`tools/data/synthesize/lhb/` 只提供 diversity-control 思路（平衡 cell、配额、去重）；frontier pipeline 没有 import、运行时依赖、代码复制或 provenance 依赖于该目录，本轮也未修改它。

| Family | Rows | Templates |
| --- | ---: | --- |
| sparse storage/compute | 160 | SP01–SP04 |
| MoE routing | 140 | MO01–MO04 |
| selective state-space | 120 | SS01–SS03 |
| attention/cache | 120 | AC01–AC04 |
| modern LLM block | 100 | LL01–LL04 |
| ragged/graph/segment | 120 | RG01–RG04 |
| quantization QDQ | 80 | QD01–QD04 |
| vision/geometry | 60 | VG01–VG03 |
| spectral/scientific | 60 | SF01–SF03 |
| retrieval/recommender | 40 | RR01–RR02 |
| **Total** | **1,000** | **35** |

Sparse family 的 forward 实际构造并消费 COO、CSR、sampled-addmm、sparse-softmax 和 semi-structured 2:4 intermediates；不是只在 metadata 标注 sparse。其他场景包括 token/expert-choice MoE、Mamba-style state carry、RoPE/GQA/sliding attention、paged KV cache、RMSNorm/SwiGLU、ragged segment/graph message passing、INT4 bitwise pack/unpack、fake FP8、grid sampling/pixel shuffle、FFT/STFT、Cholesky 和 retrieval/embedding bag。

最终 output contract 严格为一个 finite、dense、strided CUDA Tensor。Structured final output、heterogeneous structured I/O、multi-rank/distributed、backward/optimizer 和 native FP8/FP4 明确延期。

## Static 与 decontamination 证据

正式 lane：`local_artifacts/data_handoffs/prompt_tvm_v4_frontier_operator_canary1000/`。

| Item | SHA-256 / value |
| --- | --- |
| design contract | `734a13f6b8581b1414dc01b3eb5279a82dac8b32068f68e39cf9ac784a5eb85a` |
| generator | `ca053a350d46cb0474cf459ed7f2e62017fc4475a98ff28035875e3f350bb4da` |
| candidates | `566fc10aaddbf890fd140490f06a928c25594a40955c48a4d0595239f5154abf` |
| manifest | `9a215ca9032b0fbe8b5995d25a6f0c47c102d59b473b610483188d3b1f512670` |
| decisions | `e9d1c9ce52cbaa188aef0e27443676d0b7899151a0089c1e3e0185b280c853a8` |
| static summary | `6cf5ba43ca07f1ba97ccfce96b3b2d2c5d1f40869b75aab269042462da98d2be` |
| review samples | `89d77be70613daf58412a09b3e22b3995421976e25cede77f7d53a9f47cf12a7` |

Static gate 对 canonical 64,315-row root、旧 extreme-op 512-row root 和既有 semantic 1,000-row root 执行 exact-reference 与 parseable normalized-AST decontamination，并 replay registry、solver invariants、quota、typed axes、lineage 和 governance。Exact AST replay 必须固定 CPython 3.12；官方 parquet 由 `pyarrow==24.0.0` 写出，跨 writer 版本不承诺 byte-identical hash。

## A800 runtime 与 final evidence

Reference 与 liveness 都在 node22 的 8× NVIDIA A800-SXM4-80GB 上运行。环境为 Torch `2.11.0+cu129`、CUDA 12.9、cuDNN 9.17.1；KernelGym 使用 clean authority checkout `26255057463a77b23abac0f3e5eafeeebf2ebbb5`。

- reference：8×125 fresh shards，persistent train mode，5 trials、seed 42、300s timeout、64 GiB guard；1,000 executed / 1,000 passed / 0 failed；
- fresh reference-pass allowlist：1,000 ordered UUID，SHA `ee1683fc3555a6b499b756f37751f70b88b80efe84280442bcffec4c834b54f2`；
- liveness v2：8×125 fresh shards，3 trials、seeds 17/10024/20031、600s timeout、64 GiB guard；1,000 executed / 1,000 passed / 0 failed；
- liveness 检查 deterministic control/trace、input immutability、state contract、declared ATen identity 到 returned Tensor 的 taint/provenance witness，并要求 config、GPU、record final 及每个 trial 的 output/control/traced output device 全部精确为 `cuda:0`；
- liveness binding SHA `4274dedd77f9af3c2d0165a23ac1c5b1a21686c3a1b29cb9b655438da9c458a5`；validator SHA `ecaf1025d1dc3c86c48110a833981f12bd9b8f3df17c6a39747c8d6dcb5728e6`；launcher SHA `df13cea13637e5516cf3b6a58113cffe8a4530020d29e24b8b8206e88df4fec2`；analyzer SHA `1651e41f30b3510bb764b55077143fb66514fad2d5e782607cd844b97f7cb37d`。

Final analyzer exact-join 1,000 static、1,000 reference、1,000 liveness 和 1,000 accepted rows；10 families、35 templates 和全部统计轴均零掉量，failure audit 为 0。Realized ATen 只统计 `per_declared_op.matched_identities` 且 `returned_output_witness=true` 的调用，不把整张 dispatch histogram 误报为 coverage。

Declared 与 realized 是两层不同口径：`_safe_softmax`、`_scaled_dot_product_flash_attention`、`_to_sparse_semi_structured`、`cudnn_grid_sampler` 和 `rms_norm` 的 declared identity 在本次 dispatch 中 realized count 为 0，`bmm` 为 realized 75 / declared 165；实际执行由其 observed lower-level/alternative identity 闭合。下游不得把 declared inventory 当成 runtime-realized coverage。

| Final artifact | SHA-256 |
| --- | --- |
| `final/final_summary.json` | `54e218d121f10c0667b17bf93b6b39a6304365a43506812b26966ff1491c1422` |
| `final/accepted.parquet` | `a7d33f4bb5ee4494010ba1b390084e90d4cf1dd2729c224e2e809aa1c3e36d18` |
| `final/accepted.manifest.jsonl` | `0f2448d9782a195ea682103f9cde1db0e5b37d2b1d3564b8304fdff7ce2674b3` |
| `final/failure_audit.jsonl` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `final/failure_bias_report.md` | `f86cf10bf3d74f4bdd383f4a8cecdf1e4fcecd174d04d64d97b763b6f9065757` |
| `final/raw_artifact_sha256.json` | `6c77d2dcfcdf4724e4aec3d9411fbe295900200b4af76b5d7a8a42715847024f` |
| `final/runtime_evidence_samples.md` | `b4d66038c4532d1f9ffcbd5370138d1a8ab1492a8682601dbfa7a34e075bc31d` |
| `final/manual_critical_samples.md` | `d1d74c3e8eaf635d76f0fb21fe5672456c90bb85b009a59421508bcdd1cdec11` |

人工样本覆盖全部 10 families；额外 critical set 精确 join 13 条 COO/CSR/2:4、empty ragged、expert-choice MoE、paged KV、INT4、grid、FFT/STFT、embedding-bag 和 retrieval 任务。实现、commit-bound static、output-device evidence 和最终 runtime/artifact milestones 均经 `agentp --model kimi-k3-high` read-only review 为 PASS，最终 review 为 P0/P1=0；另一次独立 final artifact 复算审计为 P0/P1/P2=0。

## 诊断证据边界

`prompt_tvm_v4_frontier_operator_canary1000_pre_fix_diagnostic/` 保存 sparse constructor 缺少显式 device 时的旧 880/1,000 reference；正式修复后重新生成并 fresh 跑出上面的 1,000/1,000。`runtime/liveness/node22_authority2625505_pre_output_device_evidence_diagnostic/` 是缺少直接 output-device 字段的 390-row partial run；`diagnostics/output_device_v2_smoke.jsonl` 只证明修复门禁。它们都不进入正式 reference、liveness 或 accepted 分母，也不能复用其 hash/allowlist。

## 复现入口与未授权事项

在 clean、commit-bound worktree 中使用独立且不存在的 repro lane，依次运行：

1. `generate_frontier_operator.py`，固定 CPython 3.12、`pyarrow==24.0.0` 并传入 canonical/旧 semantic comparison artifacts；
2. `launch_reference_validation_shards.sh`，再只从该 fresh reference 的 8 shards exact-derive ordered pass UUID allowlist；
3. `launch_frontier_liveness_shards.sh candidates manifest fresh_allowlist fresh_run_dir`；
4. `analyze_frontier_run.py candidates manifest fresh_reference_dir fresh_liveness_dir empty_final_dir`。

不得覆盖正式 evidence lane，也不得把 pre-fix/partial diagnostics 当作输入。任何 generator、validator、launcher、semantic-validator closure、candidate/manifest、allowlist 或 runtime-policy hash 改变，都必须重建对应 evidence。

`training_approved=false` 和 review-only governance 位于 `final/accepted.manifest.jsonl` 与 `final/final_summary.json`，没有嵌入 `accepted.parquet` 的每一行；任何下游必须把三者作为绑定 artifact 一起消费，不能单独把 parquet 当作获批训练数据。

本 canary 没有证明任意硬件/seed、真实 fused/custom model kernel、训练收益、跨 lane policy 兼容、license/provenance merge 或数据发布。后续仍须完成 cross-lane UUID/AST 去重、provenance/license 审计、统一 runtime evidence compatibility 和等预算训练 ablation，并由人工显式设置 `training_approved=true`；本结果不自动触发扩量。
