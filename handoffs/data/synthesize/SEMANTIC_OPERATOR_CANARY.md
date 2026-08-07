# Semantic/operator single-Tensor 1k canary 交接

## 结论和范围

本 lane 不需要 LLM。1,000 条 standalone task 来自 33 个封闭、确定性的 Python template，由 registry 按固定 quota/variant 完整枚举；LLM 不能替代 source/AST replay、KernelGym reference 或运行时 ATen provenance 证明。

用户选择方案 B：本轮最终输出严格限定为单个 finite Tensor，structured final output 明确延期。1,000 条全部是 parentless semantic synthetic task，不伪造 canonical parent lineage；产物保持 `review_only`、`training_approved=false`，没有进入 `Data/`、review union 或训练。

正式入口独立位于 `tools/data/synthesize/semantic_operator_method/`：

- `generate_semantic_operator.py`：封闭 template registry、quota、static return-dependency proof、decontamination 和 exact 1k 生成；
- `validate_semantic_liveness.py`：三轮 A800 ATen dispatch、returned-output provenance、single-Tensor、exact control/state 和 registered-state proof；
- `launch_semantic_liveness_shards.sh`：8 GPU source-bound liveness launcher；
- `analyze_semantic_run.py`：精确合并 static、KernelGym reference 和 liveness，输出 review-only accepted partition。

实现 commit 为 `fd8a7cbfb24ca5d32c13f2be158c25dc8d2d6605`。四个入口的 SHA-256 分别为 generator `ff050fb136b029585f2358c588ff6dc8b359aaaf3bade56a064e4580908dab25`、validator `b0177d19377f71ae32aac13698079f858942ad9f95f866a2192bfa50f933b7ff`、launcher `5a9d26f445c759523366159ee98c9b0e9c89a9f3d430fc37819973ebc40c6d56`、analyzer `ab64990d85438b3795bc23f7be73a41e1f7cf482262456fadb2c4d2a972d3718`。

## 1k family 设计

Family 是互斥 primary quota；declared ops 是多标签，不能把两种口径相加。

| Primary family | Rows | Templates | 代表 cell |
| --- | ---: | ---: | --- |
| `attention_recurrent_stateful` | 180 | AR01–AR06，各 30 | fused SDPA、softmax attention、embedding、GRU、LSTM、stateful BN+GRU |
| `conv_norm` | 180 | CN01–CN06，各 30 | conv/transpose conv、batch/instance/group/layer norm、activation/pool |
| `heterogeneous_natural_dag` | 160 | HD01–HD05，各 32 | conv classifier、embedding attention、residual branch、index/reduction DAG |
| `matmul_reduction` | 180 | MR01–MR06，各 30 | mm/matmul/bmm、sum/mean/max/norm/clamp |
| `pool_index` | 150 | PI01–PI05，各 30 | pooling、gather/scatter/topk/index-select |
| `shape_branch` | 150 | SB01–SB05，各 30 | split/merge、pool branch、transpose/reshape branch |
| **Total** | **1000** | **33** | 所有 final output 为一个 Tensor |

Mode 分布为 stateless 758、recurrent-state 60、train-stateful 182。每个 declared op 必须由静态直线 dependency proof 连接到 return；dead op、仅增加源码长度或无法静态归因的动态调用不会进入 manifest。

AR01 的 head dimension 固定覆盖 4、8、12，各 10 行。A800 根因探针证明 6/10/14 会分解为 `_safe_softmax` + `bmm`，而 4/8/12/16 会命中 `_scaled_dot_product_efficient_attention`；本轮选取前三个已验证 cell，并把实际 fused identity 交给 liveness fail-closed 验收。

## 静态生成和去重污染

Prompt template 与第一项 decontamination root 都是 `Data/prompt_tvm_v4/train.review.parquet`，64,315 rows，SHA-256 `b07205fcadc543964cfc7ee5fd9c1e4d011f0f3481447f656e5297e40b4b99f4`。第二项 root 为 `Data/prompt_tvm_v4/synthesis/accepted/extreme_ops_v3.parquet`，512 rows，SHA-256 `57e8054f758007fc597312414da79740d72006c78b2afedf3c7d1994bd00565b`。

对两个 root 同时检查 exact reference SHA 和 Python 3.12 normalized-AST SHA；1,000 条在两个口径上均零碰撞、各自唯一。Python 3.10 与 3.12 的 `ast.dump` 不同，所以 exact replay 固定 Python 3.12；Parquet 构造和 materialization 固定 `pyarrow==24.0.0`。

| Static artifact | SHA-256 |
| --- | --- |
| `candidates.parquet` | `ddfdab60415dc325d5a7eba2dfa3615f708814322feb2713b25651f4af706ac2` |
| `manifest.jsonl` | `144d61b555539b3a6664a542f7b702dd78ed90bce4d9d07c9f403e3274725e50` |
| `decisions.jsonl` | `896828d5aa55277a62f6ffdd67aab16b8e125fc06a958396247002eee7a8e399` |
| `review_samples.md` | `ade89daf468cdecfb3c61cb7e792c69ea0c8aafe66c1b0e7ca61d9d336c239f4` |
| `summary.json` | `9653fc7d4a5a00570879093ff7d171d784fc7411145622ea80d7d975a135785b` |
| `uuid_allowlist.all1000.txt` | `b019b20f44d0a0e09dda0047d99121f46325f932b263e8b52f48eafc6e775027` |

Generator 在 manifest 中绑定生成时 Git commit；本轮生成前另行断言 HEAD 为 `fd8a7cbf...`。Static analyzer 逐行重渲染 code、declared ops 和 coverage labels，重算 reference/AST/prompt/prefix/row-payload hash，并检查 generator source SHA、quota、lineage 和治理字段；它不把 manifest 的 Git commit 与当前 HEAD 二次比较。人工读取的 12 个样本覆盖六个 family，形状公式、声明 op 和单 Tensor return 均一致。

## A800 KernelGym reference

正式 canary 在 node22 的 8× NVIDIA A800-SXM4-80GB 上运行：compute capability 8.0、Torch `2.11.0+cu129`、CUDA 12.9、cuDNN 91701。KernelGym 使用 clean authority worktree `KernelGYM-reference-authority-2625505`，commit `26255057463a77b23abac0f3e5eafeeebf2ebbb5`。

每条 standalone task 在 persistent train mode 下运行 5 trials、seed 42、180 秒逐行 timeout 和 64 GiB allocator guard。8 个 shard 各执行 125 行，`resumed=0`，结果为 1000/1000 pass；reference contract fingerprint 为 `5c406ed6b3219da23c03992b7a87264e5523d1b3fb5278f627ed1d23cd37167b`。Reference launcher SHA 为 `da63f011263caa0114d754090338304b57283002e814092b9de7ba63d46a4a38`，validator SHA 为 `691b626236b55e1b330534874a73d8f37e0a559fc35c9e788f4a2855981936fd`。

严格 verifier 重新检查 authority commit/evaluator bundle、A800/H20 allowlist、GPU/runtime、5/5 forward、train mode、memory guard、row/source hash 和单一 fingerprint。由这批新 shards 派生的 reference-pass allowlist 也是 1,000 行，SHA-256 `b019b20f44d0a0e09dda0047d99121f46325f932b263e8b52f48eafc6e775027`；它不是直接复制 full UUID inventory，虽然本轮两者集合恰好相同。

node22 在旧 generator 的首次 diagnostic reference 中出现 17 条 subprocess CUDA initialization failure；随后进行 8 卡 CUDA context 预热，旧 generator 的正式重跑和本轮 final run 都没有再出现该错误。现有 evidence 不能单独证明该瞬时故障的完整系统根因，因此只把它归为隔离的基础设施诊断，没有混入方法 failure 口径。

## A800 declared-operator liveness

1,000 个 reference-pass task 各跑 3 trials，seed 为 17、10024、20031；两个相同初始化的 persistent train-mode model 分别执行 control 与 `TorchDispatchMode` trace。每轮必须同时满足：

- final output 是一个 finite Tensor，inputs 不变，training mode 保持；
- control/trace output 与 registered state exact；
- 每个 declared op 命中 manifest 允许的 ATen identity、达到 minimum call count，并有 provenance witness 到 returned Tensor；
- dispatch sequence SHA、identity count 和 final-output op ID 完整，unknown/missing evidence fail-closed；
- 普通属性中的 Tensor 只有在对象 id 已属于 registered parameter/buffer 时才作为 alias 放行；不同 Tensor 对象即使共享 storage 仍拒绝；
- execution controls 固定 `cudnn_deterministic=true`、`cudnn_benchmark=false`、`control_trace_comparison=exact`，worker 在 `finally` 恢复进程原 flag。

8 个 shard 各执行 125 行，`resumed=0`，1000/1000 pass。Validation binding 为 `67af85077d302ccd9c7679f657b027ce0337e2727ba2e19d026c873bd8aee539`。

三条 pre-fix 根因均闭环：

- AR01 的 head_dim 4/8/12 共 30 行 × 3 trials，全部且仅命中 `aten::_scaled_dot_product_efficient_attention.<default>`，每轮一次并到达 return；
- AR04/AR05/AR06 只记录 GRU/LSTM `_flat_weights[0:4]` 的 same-object registered aliases；storage-only view 仍会作为 unregistered Tensor 拒绝；
- CN04 在 deterministic control 下 30 行 × 3 trials 的 transposed conv、cuDNN batch norm 和 tanh 都到达 return，control/trace output 与 state 全 exact；未固定 deterministic 时的根因探针曾观测到 `8.94e-8` 差异。

人工 runtime 抽检记录在 `analysis/final/runtime_evidence_samples.md`，SHA-256 `2bb8b410cdfe7ceadf2f01b34572f94c204169bfaa7898ca29791e6440c6d369`。

## Materialization 和治理

最终 review lane 为：

`local_artifacts/data_handoffs/prompt_tvm_v4_semantic_operator_canary1000`

| Final artifact | Rows / files | SHA-256 |
| --- | ---: | --- |
| `analysis/final/accepted.parquet` | 1000 | `c0466138131e53d987da1d6e72ec5e737012af3d410c2c1e6612a0bb627f1fe0` |
| `analysis/final/accepted.manifest.jsonl` | 1000 | `b9d4f16dbc9f941ab706d33492ace852f0439afb41901be0166f3e750d084a4b` |
| `analysis/final/failure_bias_report.md` | 33 templates | `f6580688db937a5ac322dde29934ffb24cd499a9bfc6787a273908ae990ead1f` |
| `analysis/final/raw_artifact_sha256.json` | 16 raw shard JSONL + candidates/manifest | `9113550ceff8f0a74a587aa23c183bbd855989fb8ed51a9cd1b52b9538d4779b` |
| `analysis/final/final_summary.json` | — | `8c94162cc12d031b5e1db5f72fc17d2461913ed6ac0519d6fe0279d8654f41c9` |

Final analyzer 报告 reference 1000/1000、liveness 1000/1000、accepted 1000；六个 family 和 33 个 template 均无选择性掉量。所有 accepted row 的 `governance_status=semantic_operator_review_only_training_not_approved`、`structured_output_deferred=true`、`training_approved=false`；final summary 另记录 `structured_output_status=explicitly_deferred`。

旧 generator 和首次失败证据整体保存在 `prompt_tvm_v4_semantic_operator_canary1000_pre_fix_diagnostic`，其 candidates SHA 为 `21e859a51735bb5b4f555e8d4661ce08401b9ffd505e56ee5aeca0d18fe9f84e`，与 final 不同；post-fix 五例 smoke 单独保存在 `prompt_tvm_v4_semantic_operator_post_fix_smoke`。Final raw hash manifest 的 16 个 shard 路径全部指向 final lane，没有复用 diagnostic shard 或 allowlist。

## Kimip milestone review

Kimip 对实现修复、新静态 1k 和最终 runtime evidence 做了只读审批，三个报告的 verdict 均为 PASS、P0=0、P1=0。最终 review 覆盖 analyzer replay、AR01 identity、CN04 exact、RNN alias、reference policy、8×125 shard 和 `resumed=0`；仓库内没有另存完整 Kimip transcript，本文件只记录 verdict 和已复核结论。

记录的 P2 均不阻塞本轮：normalized AST 依赖 Python 版本，已固定 3.12；Parquet 字节依赖 pyarrow writer，正式构造固定 24.0.0；validator 支持 resume，但本轮是全新目录且全部 `resumed=0`。

## 复现入口

以下命令要求使用一个单独的 clean worktree，checkout 到实现 commit `fd8a7cbf...`，并从新的空 output directory 开始。不要切换或覆盖当前工作目录、final lane，也不要把 `_pre_fix_diagnostic` 作为输入。示例中的 `source_worktree` 由复现者指向已准备好的 clean worktree；绝对 Data 路径用于保持最终 manifest 的 provenance root 一致。

```bash
source_worktree=/path/to/clean/slime-worktree-at-fd8a7cbf
test "$(git -C "${source_worktree}" rev-parse HEAD)" = fd8a7cbfb24ca5d32c13f2be158c25dc8d2d6605
lane_dir=/nfs/FM/chenshuailin/projects/kernel_agents/slime-v4flash-lora/local_artifacts/data_handoffs/prompt_tvm_v4_semantic_operator_canary1000_repro
test ! -e "${lane_dir}"
cd "${source_worktree}"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  uv run --python 3.12 --no-project --with 'pyarrow==24.0.0' \
  python tools/data/synthesize/semantic_operator_method/generate_semantic_operator.py \
  "${lane_dir}" \
  --template-parquet /nfs/FM/chenshuailin/projects/kernel_agents/slime-v4flash-lora/Data/prompt_tvm_v4/train.review.parquet \
  --comparison-artifact /nfs/FM/chenshuailin/projects/kernel_agents/slime-v4flash-lora/Data/prompt_tvm_v4/train.review.parquet \
  --comparison-artifact /nfs/FM/chenshuailin/projects/kernel_agents/slime-v4flash-lora/Data/prompt_tvm_v4/synthesis/accepted/extreme_ops_v3.parquet

candidate_sha=$(sha256sum "${lane_dir}/candidates.parquet" | awk '{print $1}')
test "${candidate_sha}" = ddfdab60415dc325d5a7eba2dfa3615f708814322feb2713b25651f4af706ac2
test "$(sha256sum "${lane_dir}/manifest.jsonl" | awk '{print $1}')" = 144d61b555539b3a6664a542f7b702dd78ed90bce4d9d07c9f403e3274725e50
```

在 node22、8 卡预热完成后运行 authority reference：

```bash
export LD_LIBRARY_PATH=/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-load-inline/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
SYNTH_MACHINE_COUNT=1 \
SYNTH_GPUS_PER_MACHINE=8 \
SYNTH_VIRTUAL_SHARDS_PER_GPU=1 \
SYNTH_TIMEOUT_SECONDS=180 \
SYNTH_EXPECTED_MODE_CLASS=any \
SYNTH_MAX_DEVICE_MEMORY_GIB=64 \
SYNTH_KERNELGYM_ROOT=/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-reference-authority-2625505 \
SYNTH_EXPECTED_INPUT_SHA256="${candidate_sha}" \
bash tools/data/synthesize/launch_reference_validation_shards.sh \
  0 "${lane_dir}/candidates.parquet" \
  "${lane_dir}/runtime/reference/node22_authority2625505"
```

只从这次 fresh reference shards 严格派生 allowlist；此步骤调用与 final analyzer 相同的 authority verifier，要求 1,000 个唯一 UUID/row index、当前 candidates SHA、8×125、authority commit/runtime policy 完整，并按 candidate row 顺序只写 `passed=true` 的 UUID：

```bash
ref_dir="${lane_dir}/runtime/reference/node22_authority2625505"
allowlist="${lane_dir}/analysis/reference_passed_uuids.txt"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  uv run --python 3.12 --no-project --with 'pyarrow==24.0.0' \
  python - "${lane_dir}/candidates.parquet" "${lane_dir}/manifest.jsonl" \
  "${ref_dir}" "${allowlist}" <<'PY'
from pathlib import Path
import sys

from tools.data.synthesize.semantic_operator_method.analyze_semantic_run import (
    _collect_shards,
    _verify_reference,
    _verify_static,
)

candidates, manifest, reference_dir, allowlist = map(Path, sys.argv[1:])
_, manifests, by_uuid = _verify_static(candidates, manifest)
records, bindings = _collect_shards(reference_dir)
indexed, failures = _verify_reference(records, by_uuid, candidates)
assert len(bindings) == 8 and [item["rows"] for item in bindings] == [125] * 8
assert len(records) == len(indexed) == 1000 and not failures
passed = [item["uuid"] for item in manifests if indexed[item["uuid"]].get("passed") is True]
assert len(passed) == 1000
allowlist.parent.mkdir(parents=True, exist_ok=True)
allowlist.write_text("".join(uuid + "\n" for uuid in passed), encoding="utf-8")
PY
```

Liveness 只消费上一步 fresh verifier 写出的 allowlist，随后运行 final analyzer：

```bash
live_dir="${lane_dir}/runtime/liveness/node22"
SEMANTIC_GPUS_PER_MACHINE=8 \
SEMANTIC_LIVENESS_TRIALS=3 \
SEMANTIC_LIVENESS_SEED=17 \
SEMANTIC_LIVENESS_TIMEOUT_SECONDS=600 \
bash tools/data/synthesize/semantic_operator_method/launch_semantic_liveness_shards.sh \
  "${lane_dir}/candidates.parquet" \
  "${lane_dir}/manifest.jsonl" \
  "${allowlist}" \
  "${live_dir}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  uv run --python 3.12 --no-project --with 'pyarrow==24.0.0' \
  python tools/data/synthesize/semantic_operator_method/analyze_semantic_run.py \
  "${lane_dir}/candidates.parquet" \
  "${lane_dir}/manifest.jsonl" \
  "${ref_dir}" \
  "${live_dir}" \
  "${lane_dir}/analysis/final"
```

代码检查包括 fresh pre-commit hooks（Ruff、autoflake、isort、Black）、Python 3.12 `py_compile`、shell syntax、static exact replay、A800 五例 root-cause smoke、1000-row authority reference、1000-row 三轮 liveness、final analyzer replay 和人工 raw evidence 抽查。按生产数据 pipeline 约定，本轮没有新增单元测试。

## 尚未证明的内容

本 canary 证明的是 33 个封闭 single-Tensor family 在冻结 A800 环境和有限 seeds 下的 reference correctness 与 declared-op provenance，不证明任意动态 graph、所有硬件/seed 或未来 PyTorch dispatch 都等价。Structured output、合规新 source、mode/interface 扩展、cross-lane 去重、统一 evidence policy 和训练 ablation 都仍待后续完成；在这些步骤结束并经人工批准前，不得设置 `training_approved=true`。
