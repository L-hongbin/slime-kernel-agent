# shellcheck shell=bash
# V4 task-arg assembly, extracted from full_loop_smoke.sh so it can be unit-tested
# in isolation (tests/deepseek-v4/test_dsv4_rl_task_args.py) without the cluster
# bring-up. Sets the global array TASK_ARGS from the caller's environment.
#
# Inputs (env): TASK_MODE, REWARD_MODE, PROMPT_DATA, INPUT_KEY, LABEL_KEY,
#   METADATA_KEY, MAX_CONTEXT_LEN, MAX_RESPONSE_LEN, ROLLOUT_TEMPERATURE,
#   ROLLOUT_TOP_P, ADVANTAGE_ESTIMATOR, EPS_CLIP, EPS_CLIP_HIGH, ENTROPY_COEF,
#   REPO, KERNEL_ENV_URL, KERNEL_BACKEND, USE_WANDB, WANDB_PROJECT, WANDB_GROUP,
#   DEBUG_DIR, USE_ROLLOUT_ROUTING_REPLAY, and LOAD_FORGE_ROLLOUT_DATA.
# Output: the TASK_ARGS array (declared by the caller / this function).

build_dsv4_task_args() {
  local -a routing_replay=()
  if [[ "${USE_ROLLOUT_ROUTING_REPLAY:-1}" == "1" ]]; then
    routing_replay=(--use-rollout-routing-replay)
  fi

  # Replay a captured rollout while keeping the real SGLang servers and LoRA
  # weight-update path alive.  This differs intentionally from
  # LOAD_DEBUG_ROLLOUT_DATA/DEBUG_TRAIN_ONLY, which skip SGLang entirely.  Make
  # conflicting modes fatal so a memory/update-weight canary cannot silently
  # turn into a train-only replay (or a rollout-only no-op).
  local -a forge_replay_args=()
  if [[ -n "${LOAD_FORGE_ROLLOUT_DATA:-}" ]]; then
    if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA:-}" \
          || "${DEBUG_TRAIN_ONLY:-0}" == "1" \
          || "${DEBUG_ROLLOUT_ONLY:-0}" == "1" ]]; then
      echo "FATAL: LOAD_FORGE_ROLLOUT_DATA keeps SGLang + weight sync live and cannot be combined with LOAD_DEBUG_ROLLOUT_DATA, DEBUG_TRAIN_ONLY, or DEBUG_ROLLOUT_ONLY." >&2
      return 2
    fi
    forge_replay_args=(
      --rollout-function-path slime.rollout.forge_load.generate_rollout
      --load-forge-rollout-data "${LOAD_FORGE_ROLLOUT_DATA}"
    )
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
    # Model/base, adapter/Muon, and rollout dataset state are independent
    # resume artifacts. The formal DS-V4 PP2->PP1 migration deliberately loads
    # each from a different root; omit this flag to retain slime's historical
    # --load fallback for other launchers.
    if [[ -n "${ROLLOUT_DATASET_LOAD:-}" ]]; then
      data_args+=(--rollout-dataset-load "${ROLLOUT_DATASET_LOAD}")
    fi
    if [[ "${REWARD_MODE}" == "drkernel" ]]; then
      data_args+=(--apply-chat-template-kwargs '{"enable_thinking":true}')
    else
      data_args+=(--apply-chat-template)
    fi
    # Real RL loss path (policy_loss + advantages). Borrowed from the qwen ref.
    # The formal default remains the global token reduction.  Fixed-batch
    # diagnostics may opt into the existing custom PG reducer hook while
    # disabling per-token normalization; keep both choices explicit so an A/B
    # arm cannot silently change only one half of the reduction contract.
    local -a pg_reduction_args=()
    case "${CALCULATE_PER_TOKEN_LOSS:-1}" in
      1) pg_reduction_args+=(--calculate-per-token-loss) ;;
      0) ;;
      *)
        echo "FATAL: CALCULATE_PER_TOKEN_LOSS must be 0 or 1, got '${CALCULATE_PER_TOKEN_LOSS}'." >&2
        return 2
        ;;
    esac
    if [[ -n "${CUSTOM_PG_LOSS_REDUCER_FUNCTION_PATH:-}" ]]; then
      pg_reduction_args+=(
        --custom-pg-loss-reducer-function-path
        "${CUSTOM_PG_LOSS_REDUCER_FUNCTION_PATH}"
      )
    fi
    local -a rl_args=(
      --loss-type policy_loss
      "${pg_reduction_args[@]}"
      --advantage-estimator "${ADVANTAGE_ESTIMATOR}"
      --eps-clip "${EPS_CLIP}"
      --eps-clip-high "${EPS_CLIP_HIGH}"
      --policy-loss-mode "${POLICY_LOSS_MODE:-ppo}"
      --entropy-coef "${ENTROPY_COEF}"
      # Chunked log-prob computation (qwen ref): at long ctx the last PP stage's
      # fp32 logits are ~vocab*S*4B (~8.5GB at 16k) per copy — chunking shaves
      # the peak that OOM'd the 16k train backward (v9, stage-2, 114MB short).
      --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-10000}"
    )
    # Token-capped microbatching (default OFF). The caller passes the same
    # explicit values used by slime's --max-tokens-per-gpu interfaces. PP2@16k
    # r16/r16b/r16c all OOM'd ~3GB short in the stage0
    # backward with ~16GB fragmented reserve: the heaviest packed microbatches
    # (~2 samples at mean ~7.6k tok) drive both the transient size and the
    # fragmentation. Capping tokens/microbatch halves the worst packs; a
    # single long sample still forms its own microbatch (cannot split).
    if [[ -n "${MAX_TOKENS_PER_GPU:-}" ]]; then
      rl_args+=(
        --use-dynamic-batch-size
        --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
        --log-probs-max-tokens-per-gpu "${LOG_PROBS_MAX_TOKENS_PER_GPU:-${MAX_TOKENS_PER_GPU}}"
      )
    fi
    # Behavioral old-actor + TIS. Default OFF: with both envs unset TASK_ARGS is
    # byte-identical to before. For V4 LoRA, --keep-old-actor auto-takes the
    # adapter-only path (a ~90MB snapshot swap, NOT a 2nd full model) so the PPO
    # ratio uses the policy that actually sampled the batch (real clipping) and
    # TIS = exp(megatron_old_V - sglang_V) is the pure train/infer mismatch.
    # See handoffs/deepseek-v4/lora_old_actor_tis.md for the launcher recipe.
    if [[ "${USE_KEEP_OLD_ACTOR:-0}" == "1" ]]; then
      rl_args+=(--keep-old-actor)
    fi
    if [[ "${USE_ROLLOUT_LOGPROBS:-0}" == "1" ]]; then
      # DPPO paper anchor (arXiv 2602.04879 App. C): trust region built from
      # the BEHAVIOR (rollout) policy, never the recompute. Mutually exclusive
      # with --use-tis (slime asserts; the paper is explicitly anti-naive-TIS).
      rl_args+=(--use-rollout-logprobs)
    fi
    if [[ -n "${EPS_CLIP_C:-}" ]]; then
      rl_args+=(--eps-clip-c "${EPS_CLIP_C}")
    fi
    if [[ "${POLICY_LOSS_MODE:-ppo}" == "dis" ]]; then
      rl_args+=(--dis-ratio-level "${DIS_RATIO_LEVEL:-token}")
    fi
    if [[ "${POLICY_LOSS_MODE:-ppo}" == "dppo_topk_kl_predictive" ]]; then
      rl_args+=(
        --dppo-predictive-top-k "${DPPO_PREDICTIVE_TOP_K:-20}"
        --dppo-predictive-tail-estimator "${DPPO_PREDICTIVE_TAIL_ESTIMATOR:-aggregated}"
      )
    fi
    if [[ "${USE_TIS:-0}" == "1" ]]; then
      rl_args+=(--use-tis --tis-clip "${TIS_CLIP:-2.0}" --tis-clip-low "${TIS_CLIP_LOW:-0.0}")
    fi
    local -a reward_args=()
    if [[ "${REWARD_MODE}" == "drkernel" ]]; then
      # Full DrKernel multi-turn CUDA-agent task (same custom fns as the qwen ref).
      # sequence-MIS band. Default = tight [0.999,1.001] (production). The
      # launcher may pass an explicit config to widen or disable it.
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
        --sequence-mis-config "${SEQUENCE_MIS_CONFIG:-$_seq_mis_default}"
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
      if [[ "${OVERLONG_PENALTY:-0}" == "1" ]]; then
        reward_args+=(
          --overlong-penalty
          --overlong-buffer-len "${OVERLONG_BUFFER_LEN:-2048}"
          --overlong-penalty-factor "${OVERLONG_PENALTY_FACTOR:-1.0}"
        )
      fi
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
        --wandb-always-use-train-step
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

  TASK_ARGS+=("${forge_replay_args[@]}")
  [[ "${ENTROPY_COMMON_PROBE:-0}" == "1" ]] && TASK_ARGS+=(--entropy-common-probe)
  [[ "${ASSERT_ZERO_LORA_OUT:-0}" == "1" ]] && TASK_ARGS+=(--assert-zero-lora-out)
  return 0
}
