#!/bin/bash
# ============================================================================
# DrKernel 评测示例脚本（eval-only 模式）
#
# 功能：对指定模型在 KernelBench Level-1 验证集上跑多轮评测（无训练）。
# 用法：
#   bash scripts/eval_drkernel_example.sh
#
# 可通过环境变量覆盖默认参数，例如：
#   CTX_LEN=32768 N_SAMPLES=4 TP=2 bash scripts/eval_drkernel_example.sh
# ============================================================================

set -eo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# 1. 基础参数（可通过环境变量覆盖）
# ─────────────────────────────────────────────────────────────────────────────

# 上下文窗口长度（prompt + response 的 token 总上限）
CTX_LEN=${CTX_LEN:-65536}

# 每个评测 prompt 的采样数（n_samples），越大越稳定但越慢
N_SAMPLES=${N_SAMPLES:-8}

# KernelGym 编译错误摘要的最大字符数（影响下一轮 feedback prompt 长度）
KERNELGYM_ERROR_SUMMARY_CHARS=${KERNELGYM_ERROR_SUMMARY_CHARS:-1600}

# 评测时最大生成 token 数
EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-${CTX_LEN}}

# SGLang 最大同时运行请求数（影响 decode 并发，一般不需要改）
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-96}

# Tensor Parallel 大小
TP=${TP:-2}

# ─────────────────────────────────────────────────────────────────────────────
# 2. 路径配置
# ─────────────────────────────────────────────────────────────────────────────

# 仓库根目录（自动检测）
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

# 脚本辅助目录（Ray 启动、模型配置等）
SCRIPT_HELPER_DIR=${SCRIPT_HELPER_DIR:-${REPO_ROOT}/scripts}

# 数据根目录
DATA_ROOT=${DATA_ROOT:-${REPO_ROOT}}

# ── 模型路径（必须修改为你的实际路径）──
MODEL_DIR=${MODEL_DIR:-/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B}

# ── 评测配置文件路径 ──
# YAML 格式，定义评测数据集、采样参数等，见 scripts/eval_kernelbench_level1.yaml
EVAL_CONFIG_PATH=${EVAL_CONFIG_PATH:-${SCRIPT_HELPER_DIR}/eval_kernelbench_level1.yaml}

# ── 训练数据路径（eval-only 也需要，用于初始化 data_source）──
PROMPT_DATA_PATH=${PROMPT_DATA_PATH:-${DATA_ROOT}/data/drkernel-rl-data-0513/train.parquet}

# ── KernelGym Reward Server 地址 ──
# 评测时需要一个运行中的 KernelGym 服务来编译和 benchmark kernel
RM_URL=${RM_URL:-http://192.168.16.39:20111}

# ── 目标 GPU 和编译器信息（注入到 prompt 中，告诉模型目标平台）──
DRKERNEL_GPU_NAME=${DRKERNEL_GPU_NAME:-"NVIDIA GeForce RTX 4090 (SM 8.9, Ada Lovelace)"}
DRKERNEL_COMPILER_NAME=${DRKERNEL_COMPILER_NAME:-"CUDA 12.9 (nvcc, targeting sm_89)"}

# ─────────────────────────────────────────────────────────────────────────────
# 3. 输出目录
# ─────────────────────────────────────────────────────────────────────────────

RUN_TS="$(date +%Y%m%d_%H%M%S)"
EXPT_LABEL="eval_example"
SAVE_DIR="checkpoints/${MODEL_DIR##*/}/${RUN_TS}_${EXPT_LABEL}_ctx${CTX_LEN}_n${N_SAMPLES}"
LOG_FILE="${SAVE_DIR}/run.log"
mkdir -p "${SAVE_DIR}"

# 解析 eval config YAML 中的相对路径为绝对路径
RESOLVED_EVAL_CONFIG_PATH="${SAVE_DIR}/eval_config.resolved.yaml"
sed "s|path: data/|path: ${DATA_ROOT}/data/|g" "${EVAL_CONFIG_PATH}" >"${RESOLVED_EVAL_CONFIG_PATH}"

# 日志同时输出到终端和文件
touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

export PYTHONUNBUFFERED=1

# ─────────────────────────────────────────────────────────────────────────────
# 4. 启动 Ray 集群 & 加载模型架构配置
# ─────────────────────────────────────────────────────────────────────────────

# 启动 Ray（单机 head 模式，自动检测 GPU 数量）
source "${SCRIPT_HELPER_DIR}/ray/start_cluster.sh"

# 加载模型架构参数（MODEL_ARGS 数组）
# 不同模型需要 source 不同的文件，见 scripts/models/ 目录
source "${SCRIPT_HELPER_DIR}/models/qwen3.5-27B.sh"

# ─────────────────────────────────────────────────────────────────────────────
# 5. 组装命令行参数
# ─────────────────────────────────────────────────────────────────────────────

ROLLOUT_MAX_PROMPT_LEN=$((CTX_LEN - 1))
ROLLOUT_MAX_RESPONSE_LEN=$((CTX_LEN - 1))

# ── Checkpoint 参数 ──
CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}              # HuggingFace 格式模型路径
   --ref-load ${MODEL_DIR}/torch_dist        # 参考模型（eval-only 时可指向同一模型）
   --save ${SAVE_DIR}/                       # 保存目录
   --load ${SAVE_DIR}/                       # 加载目录
   --save-interval 1
   --dist-ckpt-optim-fully-reshardable
   --distrib-optim-fully-reshardable-mem-efficient
)

# ── Rollout 参数 ──
ROLLOUT_ARGS=(
   # DrKernel 插件路径
   --custom-rm-path slime_plugins.drkernel.kernelgym_rm.custom_rm
   --rollout-function-path slime_plugins.drkernel.rollout.generate_rollout

   # 数据路径和字段映射
   --prompt-data ${PROMPT_DATA_PATH}
   --input-key ground_truth                  # parquet 中的输入字段名
   --label-key ground_truth                  # parquet 中的标签字段名
   --metadata-key extra_info                 # parquet 中的元数据字段名
   --rollout-shuffle

   # Reward 模型类型
   --rm-type deepscaler

   # ★ 关键：num-rollout=0 表示不做训练 rollout，只跑 eval
   --num-rollout 0

   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --n-samples-per-eval-prompt ${N_SAMPLES}  # 评测时每个 prompt 的采样数
   --rollout-max-prompt-len ${ROLLOUT_MAX_PROMPT_LEN}
   --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN}
   --rollout-max-context-len ${CTX_LEN}
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data

   # ★ 关键：debug-rollout-only 跳过训练，仅做 rollout + eval
   --debug-rollout-only
)

# ── 评测参数 ──
EVAL_ARGS=(
   --eval-interval 20                        # 每 20 轮 rollout 做一次 eval（eval-only 时不生效，直接触发）
   --skip-eval-before-train                  # eval-only 模式下仍需要，逻辑由 num-rollout=0 覆盖
   --eval-config "${RESOLVED_EVAL_CONFIG_PATH}"  # 评测数据集配置
   --eval-max-prompt-len ${CTX_LEN}
   --eval-max-response-len ${EVAL_MAX_RESPONSE_LEN}
   --eval-max-context-len ${CTX_LEN}
   --rm-url ${RM_URL}                        # KernelGym reward server 地址
   --dump-details ${SAVE_DIR}/dumps          # ★ 保存详细评测数据到此目录
)

# ── DrKernel 多轮参数 ──
DRKERNEL_PLUGIN_ARGS=(
   --use-multi-turn                          # 启用多轮 rollout
   --max-turns 3                             # 最大轮数
   --kernelgym-error-summary-chars ${KERNELGYM_ERROR_SUMMARY_CHARS}
)
if [ -n "${DRKERNEL_GPU_NAME}" ]; then
   DRKERNEL_PLUGIN_ARGS+=(--drkernel-gpu-name "${DRKERNEL_GPU_NAME}")
fi
if [ -n "${DRKERNEL_COMPILER_NAME}" ]; then
   DRKERNEL_PLUGIN_ARGS+=(--drkernel-compiler-name "${DRKERNEL_COMPILER_NAME}")
fi

# ── 并行和性能参数 ──
PERF_ARGS=(
   --tensor-model-parallel-size ${TP}
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

# ── GRPO 参数（eval-only 也需要，框架初始化依赖）──
GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

# ── 优化器参数（eval-only 也需要，框架初始化依赖）──
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# ── SGLang 推理引擎参数 ──
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${TP}       # 每个 SGLang 引擎占用的 GPU 数（= TP）
   --sglang-context-length ${CTX_LEN}
   --sglang-max-running-requests ${SGLANG_MAX_RUNNING_REQUESTS}
   --sglang-mem-fraction-static 0.85         # GPU 显存中分配给 KV cache 的比例
   --sglang-decode-log-interval 400
   --sglang-mamba-scheduler-strategy extra_buffer
   --router-policy consistent_hashing        # 多 engine 路由策略
   --sglang-cuda-graph-max-bs ${SGLANG_MAX_RUNNING_REQUESTS}
   --sglang-disable-custom-all-reduce
   # ── 投机解码（可选，提速约 1.2×）──
   # --sglang-speculative-algorithm EAGLE
   # --sglang-speculative-num-steps 3
   # --sglang-speculative-eagle-topk 1
   # --sglang-speculative-num-draft-tokens 4
)

# ── 其他参数 ──
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. 构建 Ray runtime 环境变量
# ─────────────────────────────────────────────────────────────────────────────

PYTORCH_CUDA_ALLOC_CONF_VALUE=${PYTORCH_CUDA_ALLOC_CONF_VALUE-expandable_segments:True}

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${REPO_ROOT}:/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SLIME_TENSOR_BACKUP_PIN_MEMORY\": \"0\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF_VALUE}\"
  }
}"

# ─────────────────────────────────────────────────────────────────────────────
# 7. 预检查：渲染 prompt 模板验证
# ─────────────────────────────────────────────────────────────────────────────

# 在启动 Ray Job 前检查 prompt 模板是否正确渲染
# 仅读取 tokenizer + chat_template，不加载模型权重
_RENDER_CHECK_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --drkernel-gpu-name "${DRKERNEL_GPU_NAME}"
   --drkernel-compiler-name "${DRKERNEL_COMPILER_NAME}"
)
if [ -n "${DRKERNEL_GPU_NAME}" ]; then
   _RENDER_CHECK_ARGS+=(--expected-gpu-words "${DRKERNEL_GPU_NAME}")
fi
PYTHONPATH="${REPO_ROOT}:${SCRIPT_HELPER_DIR}/..:${PYTHONPATH:-}" \
   python3 "${SCRIPT_HELPER_DIR}/eval_drkernel/render_prompt_check.py" "${_RENDER_CHECK_ARGS[@]}"

# ─────────────────────────────────────────────────────────────────────────────
# 8. 提交 Ray Job
# ─────────────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "评测开始"
echo "  模型: ${MODEL_DIR}"
echo "  TP: ${TP}"
echo "  上下文: ${CTX_LEN}"
echo "  采样数: ${N_SAMPLES}"
echo "  输出: ${SAVE_DIR}"
echo "  日志: ${LOG_FILE}"
echo "============================================================"

submit_ray_job --address="${RAY_JOB_ADDRESS}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-gpus-per-node 8 \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${DRKERNEL_PLUGIN_ARGS[@]}"
