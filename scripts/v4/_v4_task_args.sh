# shellcheck shell=bash
# V4 task-arg assembly, extracted from full_loop_smoke.sh so it can be unit-tested
# in isolation (tests/deepseek-v4/test_v4_rl_task_args.py) without the cluster
# bring-up. Sets the global array TASK_ARGS from the caller's environment.
#
# Inputs (env): TASK_MODE, REWARD_MODE, PROMPT_DATA, INPUT_KEY, LABEL_KEY,
#   METADATA_KEY, MAX_CONTEXT_LEN, MAX_RESPONSE_LEN, ROLLOUT_TEMPERATURE,
#   ROLLOUT_TOP_P, ADVANTAGE_ESTIMATOR, EPS_CLIP, EPS_CLIP_HIGH, ENTROPY_COEF,
#   REPO, KERNEL_ENV_URL, KERNEL_BACKEND, USE_WANDB, WANDB_PROJECT, WANDB_GROUP,
#   DEBUG_DIR, and the ROUTING_REPLAY_ARGS array.
# Output: the TASK_ARGS array (declared by the caller / this function).

build_v4_task_args() {
  local -a routing_replay=()
  if [[ "${USE_ROLLOUT_ROUTING_REPLAY:-1}" == "1" ]]; then
    routing_replay=(--use-rollout-routing-replay)
  fi

  if [[ "${TASK_MODE}" == "rl" ]]; then
    local -a data_args=(
      --prompt-data "${PROMPT_DATA}"
      --input-key "${INPUT_KEY}"
      --label-key "${LABEL_KEY}"
      --metadata-key "${METADATA_KEY}"
      --rollout-shuffle
      --rollout-max-context-len "${MAX_CONTEXT_LEN}"
      --rollout-max-response-len "${MAX_RESPONSE_LEN}"
      --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
      --balance-data
    )
    if [[ "${REWARD_MODE}" == "drkernel" ]]; then
      data_args+=(--apply-chat-template-kwargs '{"enable_thinking":true}')
    else
      data_args+=(--apply-chat-template)
    fi
    # Real RL loss path (policy_loss + advantages). Borrowed from the qwen ref.
    local -a rl_args=(
      --loss-type policy_loss
      --calculate-per-token-loss
      --advantage-estimator "${ADVANTAGE_ESTIMATOR}"
      --eps-clip "${EPS_CLIP}"
      --eps-clip-high "${EPS_CLIP_HIGH}"
      --entropy-coef "${ENTROPY_COEF}"
      # Chunked log-prob computation (qwen ref): at long ctx the last PP stage's
      # fp32 logits are ~vocab*S*4B (~8.5GB at 16k) per copy — chunking shaves
      # the peak that OOM'd the 16k train backward (v9, stage-2, 114MB short).
      --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-10000}"
    )
    local -a reward_args=()
    if [[ "${REWARD_MODE}" == "drkernel" ]]; then
      # Full DrKernel multi-turn CUDA-agent task (same custom fns as the qwen ref).
      # sequence-MIS band. Default = tight [0.999,1.001] (production). Override via
      # V4_SEQUENCE_MIS_CONFIG to widen/disable (e.g. lower=1e-9,upper=1e9 = MIS off,
      # trains on all sequences; the hook still logs mis_max_abs_log_ratio etc.).
      # Use a variable (not inline ${:-{...}}) — bash brace-matches the JSON's first
      # `}` and appends a stray one otherwise.
      local _seq_mis_default='{"aggregation":"turns_geometric","token_veto_threshold":1e-4,"lower":0.999,"upper":1.001,"use_advantage":false}'
      reward_args=(
        --custom-generate-function-path examples.kernel_agent.generate_with_cuda_agent.generate
        --custom-rm-path examples.kernel_agent.generate_with_cuda_agent.reward_func
        --custom-reward-post-process-path examples.kernel_agent.kernel_reward.reward_post_process_by_group
        --dynamic-sampling-filter-path examples.kernel_agent.kernel_filter.filter_cuda_kernel_group
        --multi-turn-prompt-config-path "${REPO}/examples/kernel_agent/prompt_config/multi_turn_cuda_kernel.yaml"
        --rollout-data-postprocess-path examples.kernel_agent.kernel_filter.sequence_mis
        --kernel-env-url "${KERNEL_ENV_URL}"
        --kernel-backend "${KERNEL_BACKEND}"
        --reference-backend torch
        --do-precheck
        --use-reference-cache
        --finalize-mode positive
        --use-multi-turn
        --filter-by-last-turn
        --padding-turns
        --max-turns "${MAX_TURNS:-1}"
        --sequence-mis-config "${V4_SEQUENCE_MIS_CONFIG:-$_seq_mis_default}"
        # REQUIRED with turns_geometric sequence-mis (slime_validate_args raises
        # without it). NOT the cause of the 2026-07-04 all-zeros step: a healthy
        # run later passed with this flag on and the same max_turns=1 — the zeros
        # were a batch-composition edge (degenerate advantages), not this flag.
        --enable-turns-dp-partitions
        --use-coverage-rs
        --coverage-rs-key time_coverage
        --coverage-rs-threshold 0.3
        --coverage-rs-factor 0.1
      )
    else
      # Gate-A isolation: exercise the V4 RL math with a reward-free stand-in so a
      # failure points at the RL path, not the KernelGym task infra.
      reward_args=(--rm-type random)
    fi
    local -a wandb_args=()
    if [[ "${USE_WANDB:-0}" == "1" ]]; then
      wandb_args=(
        --use-wandb
        --wandb-centralized
        --wandb-project "${WANDB_PROJECT}"
        --wandb-group "${WANDB_GROUP}"
        --disable-wandb-random-suffix
      )
    fi
    TASK_ARGS=(
      "${data_args[@]}"
      --rollout-temperature "${ROLLOUT_TEMPERATURE}"
      --rollout-top-p "${ROLLOUT_TOP_P}"
      "${routing_replay[@]}"
      "${rl_args[@]}"
      "${reward_args[@]}"
      "${wandb_args[@]}"
      --attention-backend flash
    )
  else
    # Original R6 SFT smoke task block (unchanged).
    TASK_ARGS=(
      --prompt-data "${PROMPT_DATA}"
      --input-key input
      --label-key label
      --metadata-key metadata
      --apply-chat-template
      --rollout-max-context-len 512
      --rollout-max-response-len 16
      --rollout-temperature "${ROLLOUT_TEMPERATURE}"
      --rollout-top-p "${ROLLOUT_TOP_P}"
      --rm-type random
      "${routing_replay[@]}"
      --loss-type sft_loss
      --calculate-per-token-loss
      --disable-compute-advantages-and-returns
      --save-debug-rollout-data "${DEBUG_DIR}/rollout_{rollout_id}.pt"
      --save-debug-train-data "${DEBUG_DIR}/train_{rollout_id}_{rank}.pt"
      --attention-backend flash
    )
  fi
}
