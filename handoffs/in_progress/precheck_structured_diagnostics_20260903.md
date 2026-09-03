# Precheck structured diagnostics and feedback-token audit (2026-09-03)

## Scope

- Worktree: `/nfs/FM/chenshuailin/projects/kernel_agents/slime-precheck-diagnostics`
- Branch: `feature/precheck-structured-diagnostics`
- Base commit: `ee230db1fff62295b14b9c0d9990ecd3a6342b6a`
- The existing first-visible-error gate and `error_message` are unchanged.
- New diagnostics expose candidate-derived facts only: a stable code, phase, and bounded evidence entries containing observed values and source locations.
- The model-facing field is singular and top-level: `precheck_diagnostic`. Its maximum container path is `root -> precheck_diagnostic -> evidence[] -> evidence item`, or four levels counting the root.
- They do not expose nearest matches, edit distances, suggested edits, repair scope, or checker internals.
- Only the local precheck is enriched. Static-check failures returned as strings by the remote KernelGym service remain unchanged rather than being reverse-parsed with error-specific templates.

## Reviewable example

For the saved Qwen3.8 `group=7`, T2 response, the existing error remains:

```text
Code precheck failed: TVM-FFI model calls are not exported: ml_forward_ops
```

The additional payload is:

```json
{
  "code": "TVM_FFI_UNRESOLVED_CALL",
  "phase": "binding_contract",
  "evidence": [
    {
      "kind": "extension_call",
      "value": "ml_forward_ops",
      "section": "MODEL_NEW",
      "line": 32,
      "column": 9,
      "snippet": "tvm_ffi_extension.ml_forward_ops(x.contiguous(), W1, b1, W2, b2, W3, b3, buf1, buf2, output)"
    },
    {
      "kind": "exported_symbol",
      "value": "mlp_forward_ops",
      "section": "APPLY_BINDINGS",
      "line": 59,
      "column": 31
    }
  ]
}
```

Feedback is serialized with `json.dumps(feedback_dict, ensure_ascii=False, default=str)` and no indentation. The Qwen3.8 tokenizer counts 115 tokens for the old rendered tool-response prompt and 265 for the new one, an increase of 150 tokens.

## Full-dump replay

Input:

- Official DeepSeek-V4-Flash-0731 KernelBench L3 evaluation
- 400 trajectories, five turns each, 2,000 saved responses
- Dump: `local_artifacts/qwen38/official_l3_5turn_eval_20260901/final_inputs/dsv4/dumps/rollout_data/eval_0.pt` in the source worktree
- Original prompt template: `multi_turn_tvm_ffi_short.yaml`
- Tokenizers loaded from the exact official Qwen3.8 and DeepSeek-V4-Flash checkpoint directories

The dump contains 244 recorded precheck failures. Local replay identifies 45 failures handled by the in-repository precheck; the other 199 are remote KernelGym static-check results and therefore receive no new payload. Four of the 45 local failures occur at T5 and would not be inserted into another model turn, leaving 41 actual new feedback messages.

### Token increase for the 41 feedback messages actually inserted into a next turn

| tokenizer | min | median | mean | p95 | max | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-27B-FP8 | 25 | 72 | 78.71 | 105 | 220 | 3,227 |
| DeepSeek-V4-Flash-0731 | 33 | 87 | 92.22 | 126 | 241 | 3,781 |

The corresponding character increase has median 250, mean 260.24, p95 364, and maximum 680.

### Mean increase by local diagnostic code

| code | next-turn feedbacks | Qwen3.8 tokens | DeepSeek tokens | maximum DeepSeek tokens |
| --- | ---: | ---: | ---: | ---: |
| `TVM_FFI_HOST_CUDA_MARKER_FORBIDDEN` | 23 | 82.83 | 98.43 | 126 |
| `MODEL_NEW_INVALID` | 6 | 68.00 | 80.00 | 80 |
| `TVM_FFI_EXTENSION_CALL_MISSING` | 6 | 25.00 | 33.00 | 33 |
| `MODEL_NEW_PYTHON_SYNTAX` | 4 | 84.50 | 96.50 | 114 |
| `TVM_FFI_UNRESOLVED_CALL` | 2 | 213.00 | 226.50 | 241 |

Across all 244 stored precheck failures, including remote-string failures and terminal T5 failures that add no next-turn prompt, the actual additions average 13.23 Qwen3.8 tokens or 15.50 DeepSeek tokens and the median is zero. Across affected trajectories, the cumulative DeepSeek increase has median 95, p95 179.60, and maximum 241 tokens.

## Precheck CPU cost

A single-process replay over the same 2,000 responses took 3.92 seconds with the old precheck and 4.60 seconds with the structured diagnostics enabled. Per response this is 1.96 ms versus 2.30 ms, an increase of 0.34 ms. This is CPU-only parsing work and does not add a compiler, GPU, or KernelGym request.

## Verification

- Replayed the old and new precheck over all 2,000 responses: zero differences in pass/fail, error type, precheck state, or `error_message`.
- `python tests/test_cuda_agent_model_feedback.py`: 34 passed.
- `python tests/test_cuda_kernel_eval.py`: 21 passed, 2 skipped (real server cases).
- `black` passes on all changed Python files.
- `ruff` was unavailable in the current environment, so its check was not run.
