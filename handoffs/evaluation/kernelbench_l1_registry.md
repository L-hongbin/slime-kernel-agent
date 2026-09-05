# KernelBench L1 模型精度登记

本文件是 KernelBench Level 1（`tvm_ffi`、KernelGym）的统一结果登记。`compile` 为编译通过率，`correct` 为数值正确率，`fast@1.0/1.2` 还要求相对 torch 参考达到对应 speedup；评测指标均由 `summarize_eval.py` 按 trajectory 汇总。

## 汇总

| 模型 | checkpoint | backend / ctx=resp / turns / n | compile | correct | fast@1.0 | fast@1.2 |
|---|---|---|---:|---:|---:|---:|
| Qwen3.8-27B（DataV4，新 reward，T=1.0） | step 80 (`iter_0000079`) | tvm_ffi / 24576 / 1 / 8 | 96.25% | 84.88% | 18.12% | 14.88% |
| Qwen3.8-27B（DataV4，predictive-DPPO，T=1.1） | step 80 (`iter_0000079`) | tvm_ffi / 16384 / 1 / 8 | 96.88% | 87.25% | 20.50% | 15.12% |
| Qwen3.8-27B（DataV4，predictive-DPPO，T=1.1） | step 100 (`iter_0000099`) | tvm_ffi / 16384 / 1 / 8 | 97.88% | 90.12% | 21.38% | 16.88% |
| Qwen3.8-27B（DataV2，predictive-DPPO，T=1.1） | step 40 (`iter_0000039`) | tvm_ffi / 16384 / 1 / 8 | 88.75% | 74.50% | 18.75% | 14.88% |
| Qwen3.8-27B（DataV2，predictive-DPPO，T=1.1） | step 80 (`iter_0000079`) | tvm_ffi / 16384 / 1 / 8 | 84.12% | 78.25% | 23.62% | 19.38% |
| Qwen3.6-27B（slime RL，full-async） | iter 39 HF | tvm_ffi / 16384 / 1 / 8 | 59.75% | 35.00% | 5.62% | 4.38% |
| Step-3.7-Flash（language-only） | base | tvm_ffi / 32768 / 3-best / 8 / high+MTP | 31.25% | 18.88% | 5.12% | 4.12% |
| gpt-oss-120b（MXFP4） | base @H20 TP8 | tvm_ffi / 32768 / 3-best / 8 | 53.87% | 42.88% | 14.12% | 8.00% |
| DeepSeek-V4-Flash（FP8） | base @H20 TP4+MTP+marlin | tvm_ffi / 32768 / 3-best / 8 | 86.00% | 65.12% | 29.12% | 14.25% |
| DeepSeek-V4-Flash（r21 rsLoRA，predictive-DPPO） | step 720 (`iter719` adapter) @H20 | tvm_ffi / 12288 / 1 / 8 | 97.88% | 90.25% | 21.38% | 18.38% |
| Nemotron-3-Super-120B-A12B-FP8 | base @H20 TP8+EP8 | tvm_ffi / 32768 / 3-best / 8 | 21.00% | 7.62% | 2.50% | 1.62% |

单轮 Qwen/LoRA 与三轮 best-of-3 基座结果不可直接横比。DataV4 step 80/100、DataV2 step 40/80 与新 reward step80 也不是同一训练 lineage；完整 L1/L2/L3 paired 对比和证据哈希见 `handoffs/qwen38/kernelbench_eval.md`。新 reward 行中的 `24576` 是 prompt+response 总 context cap，实际 response budget 为 `24576-prompt_tokens`。DeepSeek step 720 的 800 条原始 dump 中有 1 条通过 formal precheck 后遇到 KernelGym 断连，因此四项比例各有最多 0.125 个百分点的不确定宽度。

## Qwen3.6-27B iter 39

单轮 800 trajectories 得到 compile/correct/fast@1.0/fast@1.2 = 59.75/35.00/5.62/4.38%，说明 RL 已学会 TVM-FFI 格式，但性能仍是瓶颈。该 full-async lineage 约 iter 89 因磁盘写满终止，iter 79 的异步 checkpoint 损坏，iter 39 是最后一个完整 checkpoint；HF 转换包含 11 个 safetensors、866 tensors、54.6 GiB。评测在 node69 的 8×H20 上运行两个 TP4 engine，配置为单轮、16K、每题 8 样本。`spec_accept_rate=0` 只表示该次 EAGLE 没有提速，不影响四项正确性指标。

证据：`experiments/EvalFAsync.tvm_ffi.Qwen3.6-27B.CTX16384/iter_39/`；原登记对应 `summary.20260622.122238.txt`。

## Step-3.7-Flash

| effort | compile T1/T2/T3/best | correct T1/T2/T3/best | fast@1.0 best | fast@1.2 best |
|---|---|---|---:|---:|
| high | 1.00 / 14.62 / 22.50 / 31.25 | 0.62 / 7.38 / 13.12 / 18.88 | 5.12 | 4.12 |
| medium | 2.38 / 15.62 / 21.88 / 30.38 | 1.12 / 8.75 / 11.62 / 17.88 | 4.88 | 3.75 |
| low | 2.50 / 15.25 / 23.62 / 32.00 | 0.62 / 8.75 / 11.62 / 18.12 | 5.25 | 3.62 |

三档成绩处于同一噪声区间，主要失败是未训练基座不熟悉 TVM-FFI 绑定，而不是 effort 设置。评测使用 node64 8×H20、language-only TP8、三轮 32K、每题 8 样本；tokenizer 必须设置 `fix_mistral_regex=True`，否则 transformers v5 会丢失空白。证据目录为 `experiments/Eval.TVMFFI.Step3.7-Flash.LangOnly.*`。

## gpt-oss-120b

| 指标 | T1 | T2 | T3 | best |
|---|---:|---:|---:|---:|
| compile | 17.75 | 35.50 | 41.88 | 53.87 |
| correct | 11.12 | 23.75 | 29.75 | 42.88 |
| fast@1.0 | 3.25 | 6.88 | 11.25 | 14.12 |
| fast@1.2 | 2.12 | 4.00 | 6.25 | 8.00 |

MXFP4 要求 Hopper；A800 无可用性能，H20 上必须使用 TP8 单 engine，TP4 双 engine 曾触发 harmony encoding 并发 race。多轮 assistant 历史还必须先抽取 final channel。可信结果来自 node164 8×H20、medium effort、三轮 32K、每题 8 样本，证据为 `experiments/Eval.TVMFFI.gpt-oss-120b.*/summary.20260628.111822.txt`。

## DeepSeek-V4-Flash base

| 指标 | T1 | T2 | T3 | best |
|---|---:|---:|---:|---:|
| compile | 39.00 | 64.75 | 75.88 | 86.00 |
| correct | 27.00 | 37.50 | 48.75 | 65.12 |
| fast@1.0 | 8.25 | 14.25 | 23.25 | 29.12 |
| fast@1.2 | 5.50 | 6.62 | 12.12 | 14.25 |

这次部署中默认 Triton fused-MoE 与权重形状不兼容；H20 使用 `--sglang-moe-runner-backend marlin` 后跑通。所用 cluster top-k 与 DeepGEMM HC-prenorm 路径使当时的 A800 尝试失败，这不是对其他实现的通用硬件支持结论。评测为 node164 8×H20、TP4+MTP、thinking-on、三轮 32K、每题 8 样本；证据为 `experiments/Eval.TVMFFI.deepseek-v4-flash.*/summary.20260628.123236.txt`。

## Nemotron-3-Super-120B-A12B-FP8

| 指标 | T1 | T2 | T3 | best |
|---|---:|---:|---:|---:|
| compile | 0.75 | 10.62 | 14.75 | 21.00 |
| correct | 0.25 | 3.38 | 5.62 | 7.62 |
| fast@1.0 | 0.00 | 1.00 | 1.88 | 2.50 |
| fast@1.2 | 0.00 | 0.62 | 1.25 | 1.62 |

800/800 trajectories 完成且无 missing `env_result`；候选 kernel 失败日志不代表作业失败。评测使用 node170 8×H20、TP8/EP8、FP8 KV、FlashInfer attention、Triton MoE、no-buffer Mamba 和 CUDA Graph；证据为 `experiments/Eval.TVMFFI.nemotron-3-super-120b.*/summary.20260629.121054.txt`。

DeepSeek-V4-Flash r21 rsLoRA 的 step 0–720 曲线、bootstrap 诊断和 H20/H200 边界由 `handoffs/deepseek-v4/kernelbench_l1_lora_curve_20260724.md` 单独维护。
