# DeepSeek-V4 packed-MXFP4/W4A16 contract

本文维护官方 mixed checkpoint 中 routed experts 的存储、转换、trainer compute 与 rollout compute 契约。并行分片见 `dsv4_megatron_sharding_contract.md`，LoRA checkpoint 与 serving 见 `lora_training_features.md` 和 `lora_serve_design.md`，MTP/DSpark 见 `mtp_speculative_decoding.md`，跨引擎概率差异见 `train_rollout_mismatch.md`

## 工程结论

- 官方 checkpoint 的 routed experts 保持 packed MXFP4 驻留，不在转换或加载阶段展开成 BF16 参数
- Trainer 对每个被路由到的 frozen expert 临时解包为 BF16，执行 W4A16 forward，并在 backward 只计算 `dX`
- Rollout 使用 DSpark 的 `flashinfer_mxfp4` SM90 runner，`a2a=none`，同时保留 dp-attention 和 attention TP1
- Trainer 与 rollout 必须同时启用 packed expert mode；任何单侧降级都会 fail closed
- Checkpoint family 必须按 routed-expert tensor dtype 判断；官方 mixed checkpoint 与 uniform-FP8 checkpoint 的 config/index 不能可靠区分 family
- 当前 DSpark packed KV pool 只支持 `fp8_e4m3` KV，formal launcher 对 bf16 KV fail closed
- Runtime identity 是固定 DSpark base、ordered patch series、容器依赖和 runtime fingerprint 的组合，不能用未验收的 in-place upgrade 替换

## Native 与 Megatron 存储格式

设 hidden size `H=4096`，expert intermediate size `I=2048`

| Tensor | Native logical shape | Native packed shape | Megatron local buffer |
|---|---:|---:|---:|
| `w1` / `w3` | `[I,H]` | `[I,H/2]` I8 + `[I,H/32]` E8M0 | 合并为 `gate_up_proj_fp4 [E,2I,H/2]` 与 `gate_up_proj_sf [E,2I,H/32]` |
| `w2` | `[H,I]` | `[H,I/2]` I8 + `[H,I/32]` E8M0 | `down_proj_fp4 [E,H,I/2]` 与 `down_proj_sf [E,H,I/32]` |

每个 weight byte 包含两个 E2M1 值，low nibble 对应偶数 K，high nibble 对应奇数 K。E2M1 magnitude 为 `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`，scale 每 32 个连续 K 元素一个 E8M0 byte，数值为 `2^(byte-127)`

`w1` 与 `w3` 按 gate/up 顺序沿 output 轴拼接。SGLang runner 的内部 reorder 属于 runner private layout，不能写回 Megatron checkpoint

除 routed experts 外，官方 checkpoint 还含 FP8 weights 与 E8M0 scales。Converter 按 tensor dtype 解码这些 scales；不能把 E8M0 byte 当成普通整数或 F32 scale

## Trainer contract

`V4_FP4_FROZEN_EXPERTS=1` 必须在模型构建前设置。`V4GroupedExperts` 随后注册四个 persistent `uint8` buffers，不创建 routed-expert Parameters

### Forward 与 backward

1. `_unpack_mxfp4` 用 E2M1 LUT 和 PyTorch native `float8_e8m0fnu` cast 生成临时 BF16 weight
2. `_FrozenFp4ExpertLinear.forward` 执行 BF16 activation × BF16 decoded weight
3. Autograd context 只保存 packed weight 与 scale buffers，不保存展开后的 BF16 weight
4. Backward 再次临时解包，只返回 `grad_x = grad_out @ W`
5. `forward_dispatched` 按 local expert 的连续 token segment 顺序处理，一次只保留一个 expert 的展开权重

BF16 decode 对合法 E2M1 magnitude 与 power-of-two scale 是精确表示。`0x00` scale 表示未加载或无效 checkpoint，`0xFF` 产生 NaN scale；`verify_fp4_loaded` 对任意一个边界 byte 都拒绝启动

V4 的 clamped SwiGLU 仍由 `V4GroupedExperts._apply_gate` 执行，不能用缺少 clamp 的 generic grouped MLP 替换

### Distributed checkpoint

- Packed buffers 沿 global expert axis 做 EP sharding
- CP/DP replicas 通过 expert-DP `replica_id` 标记，不能让两个 replica 同时声明 main shard
- Base conversion 和 cold load 包含 packed buffers
- LoRA adapter-only save 按 trainable parameters 过滤，不重复保存 frozen packed base
- Packed experts 不进入 optimizer、DDP grad buffer 或 weight sync payload

## Conversion contract

正式转换入口是 `scripts/dsv4/convert_torch_dist.sh`

```bash
FP4_EXPERTS=1 \
INCLUDE_MTP=0 \
PP_SIZE=<pp> EP_SIZE=8 \
FIRST_LAYERS=<first> LAST_LAYERS=<last> \
bash scripts/dsv4/convert_torch_dist.sh
```

调用方必须显式提供与目标 topology 一致的 `NNODES`、`NODE_RANK`、`MASTER_ADDR`、`CHECKPOINT` 和 `SAVE`。`NNODES * NPROC_PER_NODE` 必须等于 `PP_SIZE * EP_SIZE`

`FP4_EXPERTS=1` 的转换语义为

- routed expert weight 与 scale bytes 原样写入 torch_dist
- `w1/w3` 只做确定性的 gate/up 拼接
- non-expert FP8 tensor 按目标模块要求转换，E8M0 scale 由 dtype 驱动解码
- 输出模型以 packed mode 构建，checkpoint metadata 保留 EP sharding

转换完成后必须由新进程 cold-read 输出并运行 `verify_torch_dist.py`。当前脚本会 hard-fail

- 请求 key 不在 torch_dist metadata，防止 DCP 对错误 namespace 静默 no-op
- Packed buffer resident dtype 不是 `uint8`
- 每个 rank 的代表层 `model.layers[0]` 中任一 local expert scale buffer 含 `0x00/0xFF`

Verifier report 还记录以下项目，但当前没有对其设 threshold，也不会仅因非零而退出。接受 conversion artifact 前必须由外部 report audit 明确检查

- `load_state_dict_missing=[]` 与 `load_state_dict_unexpected=[]`
- 所有 rank 报告的 global expert range 连续覆盖 `0..255`
- 每个 rank 代表层的 first/last local expert 四组 packed tensor byte diff 为零
- 每个 rank 代表层的 `q_a_proj_diff=0`

Trainer cold load 随后对每个 packed `V4GroupedExperts` module 调用 `verify_fp4_loaded`，因此正式加载会检查全部层的 local scale buffers。Conversion verifier 的代表层检查不能替代这个 post-load gate

`INCLUDE_MTP=0` 是 trainer 不训练 MTP 时的默认值。启用 MTP conversion 需要独立验证 `mtp.*`，不能把 non-MTP verifier 结果外推到 MTP subtree

## Rollout contract

Packed W4A16 rollout 的关键配置为

| Setting | Contract |
|---|---|
| `V4_FP4_FROZEN_EXPERTS=1` | Trainer 构建 packed buffers |
| `SGLANG_DSV4_FP4_EXPERTS=1` | Rollout 明确选择 packed expert family |
| `USE_SGLANG_DEEPEP=0` | 避免进入 W4A8 DeepGEMM path |
| `--sglang-moe-a2a-backend none` | 使用 W4A16 runner 支持的 dispatch |
| `--sglang-moe-runner-backend flashinfer_mxfp4` | 避免 `auto` 落入 FP8 method |
| `--sglang-enable-dp-attention` | 保持 attention TP1，兼容未切分 LoRA adapter |
| `SGLANG_SHARED_EXPERT_TP1=1` | Shared expert 与 unsharded LoRA adapter 对齐 |
| `--sglang-kv-cache-dtype fp8_e4m3` | 满足 DSpark packed uint8 KV pool |

正式配置由 `scripts/dsv4/run.deepseek_v4_flash.fp4.formal.rl.sh` 提供，共享门禁由 `_dsv4_launch_core.sh` 维护。Speculative decoding、draft weights 和 routing replay 的额外约束由 `mtp_speculative_decoding.md` 维护

Rollout 的模型目录可以包含 trainer 不消费的 DSpark draft subtree，但 base tensor、tokenizer、chat template、LoRA mapping 和 weight version 必须与 trainer contract 对齐

## Fail-closed gates

- Packed mode + non-I8/U8 routed expert source：拒绝
- Packed expert source + packed mode disabled：拒绝
- Trainer packed mode + rollout packed mode disabled：拒绝
- Packed mode + DeepEP a2a：拒绝
- Packed mode + non-W4A16 runner：不得作为正式配置
- DSpark runtime + bf16 KV：拒绝
- Packed mode enabled 但模型中没有 packed `V4GroupedExperts` module：拒绝
- Invalid E8M0 scale byte、缺失 scale tensor、错误 tensor shape：拒绝
- Topology 与 torch_dist sharding metadata 不匹配：拒绝

这些门禁只验证 family 与结构一致性。Probability parity 仍需 paired full-vocabulary probe，不能用 checkpoint byte identity 代替

## 验证入口

| 层 | 入口 |
|---|---|
| E2M1/E8M0 decode、forward、`dX`、dispatcher、EP metadata | `tests/deepseek-v4/test_dsv4_fp4_frozen_experts.py` |
| Native checkpoint mapping | `custom_kernels/deepseek_v4/megatron/native_checkpoint.py` |
| Conversion 与 cold verification | `scripts/dsv4/convert_torch_dist.sh`、`custom_kernels/deepseek_v4/megatron/verify_torch_dist.py` |
| Checkpoint-family preflight | `scripts/dsv4/probe_expert_dtype.py`、`scripts/dsv4/_dsv4_cluster_lib.sh` |
| Trainer post-load guard | `V4GroupedExperts.verify_fp4_loaded` |
| Rollout launch gates | `scripts/dsv4/_dsv4_launch_core.sh` |
| Formal lifecycle | `scripts/dsv4/launch_formal_managed.sh` |
| DSpark source identity | `scripts/dsv4/patches/dspark_port_series/` |

## Unsupported alternatives

- Full-bf16 KV patch 只属于支持该 layout 的旧 fleet runtime；当前 DSpark packed pool 不支持，formal launcher 明确拒绝。相关 MTP 性能边界由 `mtp_speculative_decoding.md` 保留
- Batched full-layer expert unpack 会制造过大的临时 BF16 stack，当前 trainer 保持 per-expert transient discipline
- FP4 weight 展开成常驻 BF16 或 FP8 会失去 packed residency 和显存收益，不属于当前 contract
- SGLang runtime 更新必须重放 ordered patch series、重建容器并重新跑 GPU gates；旧版升级可行性清单不作为当前运行依据
- DPPO recipe、entropy、filter 和 run lineage 属于训练专题文档，不在本存储/compute contract 重复维护
