"""Human-reviewed operation adapters for two fixed reference programs.

This is an annotated pilot, not an automatic arbitrary-program semantic mapper.
Function names below are reviewed source locators; labels are not assigned by
name similarity. Unsupported or missing bodies fail closed.
"""

import re

from .semantic_state import (
    bind_scalar_arguments,
    call_sites,
    check_gemm_indices,
    check_mask_stride,
    component,
    evidence,
    gemm_descriptor,
    integer_locals_before_call,
    node,
)

PILOT = {"qwen38": [0, 7, 348, 351], "dsv4": [0, 3, 348, 349]}


def body(turn, name, section="CUDA_KERNELS"):
    return component(turn, section, name)


def anchor(owner, target, ordinal=0):
    calls = call_sites(owner, target)
    if ordinal >= len(calls):
        raise ValueError(f"missing reviewed call {owner['name']}:{target}[{ordinal}]")
    return evidence(owner, calls[ordinal])


def checked_gemm(owner, ordinal, values, roles, operation, shape, batched=False):
    target = "cublasSgemmStridedBatched" if batched else "cublasSgemm"
    call = call_sites(owner, target)[ordinal]
    actual_values = integer_locals_before_call(owner, call, values)
    descriptor = gemm_descriptor(call, actual_values, roles)
    return {
        "gemm_descriptor": descriptor,
        "index_contract": check_gemm_indices(
            descriptor, operation, *shape, expected_batch=128 * 8 if batched else None
        ),
        "gemm_evidence": evidence(owner, call),
    }


def mlp_nodes(turn, model, group):
    nodes = []
    sizes = [16384, 16384, 16384, 8192]
    for layer in range(3):
        K, N, M = sizes[layer], sizes[layer + 1], 128
        values = dict(
            batch=M,
            in_f=K,
            out_f=N,
            in_features=K,
            out_features=N,
            input_size=16384,
            hidden1=16384,
            hidden2=16384,
            output_size=8192,
        )
        if model == "dsv4" and group == 0:
            owner = body(turn, "mlp_forward_launcher")
            roles = {
                "input" if layer == 0 else f"tmp{layer}": "X",
                f"w{layer+1}": "W",
                "output" if layer == 2 else f"tmp{layer+1}": "Y",
            }
            detail = checked_gemm(owner, layer, values, roles, "linear", (M, N, K))
            sources = [detail["gemm_evidence"], anchor(owner, "launch_add_bias_relu", layer)]
            method, relu = "cublas_gemm_then_bias_activation_kernel", "bias_activation_kernel"
        else:
            name = (
                "linear_forward_launcher"
                if model == "dsv4"
                else (
                    "mlp_linear_forward"
                    if group == 0
                    else "gemm_bias_launcher" if layer == 2 else "gemm_bias_relu_launcher"
                )
            )
            owner = body(turn, name)
            calls = call_sites(owner, "cublasSgemm")
            if calls:
                roles = {"input": "X", "weight": "W", "output": "Y", "x": "X", "W": "W", "out": "Y"}
                detail = checked_gemm(owner, 0, values, roles, "linear", (M, N, K))
                sources = [evidence(owner), detail["gemm_evidence"]]
                method, relu = "cublas_gemm_then_bias_activation_kernel", "bias_activation_kernel"
            elif call_sites(owner, "cublasLtMatmul"):
                detail = {"index_contract": {"status": "unknown", "reason": "Lt_descriptor_not_interpreted"}}
                sources = [evidence(owner)]
                method, relu = "cublasLt_bias_epilogue", "separate_relu_kernel"
            elif call_sites(owner, "cublasGemmEx"):
                detail = {"index_contract": {"status": "unknown", "reason": "GemmEx_descriptor_not_interpreted"}}
                sources = [evidence(owner)]
                method, relu = "cublasGemmEx_TF32_then_bias_activation_kernel", "bias_activation_kernel"
            elif call_sites(owner, "gemm_bias_act_kernel"):
                detail = {"index_contract": {"status": "unknown", "reason": "custom_kernel_not_proven"}}
                sources = [evidence(owner), evidence(body(turn, "gemm_bias_act_kernel"))]
                method, relu = "custom_tiled_gemm_bias_activation_epilogue", "gemm_epilogue"
            else:
                raise ValueError(f"unreviewed MLP path: {model}/{group}/T{turn['turn_idx']+1}")
        input_role = "input" if layer == 0 else f"hidden{layer-1}"
        output_role = "output" if layer == 2 else f"preactivation{layer}"
        nodes.append(
            node(
                f"linear{layer}",
                method,
                [input_role, f"weight{layer}", f"bias{layer}"],
                output_role,
                sources,
                logical_shape=[M, N, K],
                candidate_activation_inputs=[input_role],
                candidate_parameter_inputs=[f"bias{layer}", f"weight{layer}"],
                wiring_origin="human_reviewed_forward_loop_or_native_chain",
                **detail,
            )
        )
        if layer < 2:
            nodes.append(
                node(
                    f"relu{layer}",
                    relu,
                    [output_role],
                    f"hidden{layer}",
                    sources,
                    candidate_activation_inputs=[output_role],
                    candidate_parameter_inputs=[],
                    wiring_origin="human_reviewed_activation_flag_and_epilogue",
                    materialization="may_be_fused_not_a_separate_buffer",
                )
            )
    return nodes


def attention_wire_trace(nodes, model, group):
    """Track value versions in reviewed driver call operands; no pointer alias inference.

    Initial parameter roles are manually aligned to the reference. Intermediate
    names are learned from writes, so reading before a mapped write stays unknown.
    This does not execute branches or prove the kernel bodies implement each role.
    """
    roles = {"x": "input", "x_ptr": "input"}
    parameter_names = {
        "ln1_weight": ["ln1_w", "ln1_w_ptr"],
        "ln1_bias": ["ln1_b", "ln1_b_ptr"],
        "ln2_weight": ["ln2_w", "ln2_w_ptr"],
        "ln2_bias": ["ln2_b", "ln2_b_ptr"],
        "qkv_weight": ["attn_qkv_w", "c_attn_w", "c_attn_w_ptr"],
        "qkv_bias": ["attn_qkv_b", "c_attn_b", "c_attn_b_ptr"],
        "attention_weight": ["attn_proj_w", "c_proj_w", "c_proj_attn_w_ptr"],
        "attention_bias": ["attn_proj_b", "c_proj_b", "c_proj_attn_b_ptr"],
        "fc_weight": ["mlp_fc_w", "c_fc_w", "c_fc_w_ptr"],
        "fc_bias": ["mlp_fc_b", "c_fc_b", "c_fc_b_ptr"],
        "projection_weight": ["mlp_proj_w", "mlp_cproj_w", "c_proj_mlp_w", "c_proj_mlp_w_ptr"],
        "projection_bias": ["mlp_proj_b", "mlp_cproj_b", "c_proj_mlp_b", "c_proj_mlp_b_ptr"],
    }
    for role, names in parameter_names.items():
        roles.update({name: role for name in names})
    # These explicit transpose buffers are reviewed in the dsv4/349 driver.
    if model == "dsv4" and group == 349:
        roles.update(
            d_w_attn_t="qkv_weight_transposed",
            d_w_c_proj_t="attention_weight_transposed",
            d_w_c_fc_t="fc_weight_transposed",
            d_w_mlp_proj_t="projection_weight_transposed",
        )
    for n in nodes:
        invocation = n["evidence"][0]["source"]
        args = invocation["arguments"]
        target = invocation["target"]
        slot = n["slot"]
        if target == "cublasSgemm":
            reads, writes = [7, 9], [12]
        elif target == "cublasSgemmStridedBatched":
            reads, writes = [7, 10], [14]
        elif target == "gemm_batched_kernel":
            reads, writes = [0, 1], [2]
        elif slot in {"ln1", "ln2"} or target == "launch_gemm_bias":
            reads, writes = [0, 1, 2], [3]
        elif slot == "split_heads":
            reads, writes = [0], [1, 2, 3]
        elif slot in {"qk", "pv"}:
            reads, writes = [0, 1], [2]
        elif slot == "causal_softmax":
            reads, writes = [0], [0]
        elif slot == "gelu":
            reads, writes = [0], [1] if model == "dsv4" else [0]
        elif slot == "merge_heads":
            reads, writes = [0], [1]
        elif slot.startswith("residual"):
            if target == "fused_add_kernel":
                reads, writes = [1, 2], [0]
            elif model == "qwen38" and group == 351:
                reads, writes = [0, 1], [0]
            else:
                reads, writes = [0, 1], [2]
        else:
            raise ValueError(f"unmapped driver operands at {slot}")
        before = [roles.get(args[i], "unknown_value_before_mapped_write") for i in reads]
        n["candidate_activation_inputs"] = sorted(r for r in before if "weight" not in r and "bias" not in r)
        parameters = [r for r in before if "weight" in r or "bias" in r]
        if "bias_evidence" in n and not (model == "qwen38" and group == 348):
            bias = n["bias_evidence"]["source"]["arguments"][1]
            parameters.append(roles.get(bias, "unknown_parameter"))
        n["candidate_parameter_inputs"] = sorted(parameters)
        if "unknown_value_before_mapped_write" in before:
            n["candidate_activation_inputs"] = None
        n["candidate_wiring"] = {
            "read_buffers": [args[i] for i in reads],
            "read_value_roles": before,
            "write_buffers": [args[i] for i in writes],
            "overwrites_value_roles": [roles.get(args[i], "new_buffer") for i in writes],
            "read_write_alias_buffers": sorted({args[i] for i in reads} & {args[i] for i in writes}),
            "scope": "selected_driver_calls_with_manual_parameter_roles",
        }
        output = n["reference_output"]
        outputs = [output] if isinstance(output, str) else output
        for index, value in zip(writes, outputs, strict=True):
            roles[args[index]] = value
        if model == "dsv4" and group == 349 and slot == "split_heads":
            roles[args[4]] = "k_heads_transposed"
        if model == "qwen38" and group == 351 and slot == "residual1" and target == "add_kernel":
            n["candidate_wiring"]["scope"] = "first_call_only; later_memset_and_adds_require_manual_review"
            n["candidate_activation_inputs"] = None


ATTENTION_GRAPH = [
    ("ln1", ["input", "ln1_weight", "ln1_bias"], "norm1"),
    ("qkv", ["norm1", "qkv_weight", "qkv_bias"], "qkv_packed"),
    ("split_heads", ["qkv_packed"], ["q_heads", "k_heads", "v_heads"]),
    ("qk", ["q_heads", "k_heads"], "scores"),
    ("causal_softmax", ["scores"], "probabilities"),
    ("pv", ["probabilities", "v_heads"], "head_output"),
    ("merge_heads", ["head_output"], "attention_output"),
    ("attention_projection", ["attention_output", "attention_weight", "attention_bias"], "projected_attention"),
    ("residual1", ["input", "projected_attention"], "residual1_output"),
    ("ln2", ["residual1_output", "ln2_weight", "ln2_bias"], "norm2"),
    ("mlp_fc", ["norm2", "fc_weight", "fc_bias"], "mlp_hidden"),
    ("gelu", ["mlp_hidden"], "mlp_activated"),
    ("mlp_projection", ["mlp_activated", "projection_weight", "projection_bias"], "mlp_output"),
    ("residual2", ["residual1_output", "mlp_output"], "output"),
]


def attention_nodes(turn, model, group):
    # Source-to-reference mapping was inspected for each selected trajectory.
    if model == "qwen38" and group == 348:
        owner = body(turn, "transformer_block_forward", "APPLY_BINDINGS")
        targets = [
            "launch_layernorm",
            "launch_gemm_bias",
            "launch_qkv_rearrange",
            "launch_batched_qk",
            "launch_causal_softmax",
            "launch_batched_sv",
            "launch_attn_rearrange",
            "launch_gemm_bias",
            "launch_residual_add",
            "launch_layernorm",
            "launch_gemm_bias",
            "launch_gelu",
            "launch_gemm_bias",
            "launch_residual_add",
        ]
    elif model == "qwen38":
        owner = body(turn, "transformer_block_launcher")
        targets = [
            "layernorm_kernel",
            "cublasSgemm",
            "qkv_split_kernel",
            "cublasSgemmStridedBatched",
            "causal_softmax_kernel",
            "cublasSgemmStridedBatched",
            "reshape_attn_out_kernel",
            "cublasSgemm",
            "add_kernel" if turn["turn_idx"] == 1 else "fused_add_kernel",
            "layernorm_kernel",
            "cublasSgemm",
            "gelu_kernel",
            "cublasSgemm",
            "add_kernel",
        ]
    elif group == 348:
        owner = body(turn, "transformer_block_forward")
        targets = [
            "layernorm_kernel",
            "cublasSgemm",
            "split_qkv_kernel",
            "qk_matmul_kernel",
            "softmax_kernel",
            "attn_v_matmul_kernel",
            "rearrange_attn_kernel",
            "cublasSgemm",
            "add_residual_kernel",
            "layernorm_kernel",
            "cublasSgemm",
            "gelu_kernel",
            "cublasSgemm",
            "add_residual_kernel",
        ]
    else:
        owner = body(turn, "transformer_block_forward")
        targets = [
            "layernorm_kernel",
            "gemm_batched_kernel",
            "qkv_split_permute_kernel",
            "gemm_batched_kernel",
            "causal_softmax_kernel",
            "gemm_batched_kernel",
            "merge_heads_kernel",
            "gemm_batched_kernel",
            "add_kernel",
            "layernorm_kernel",
            "gemm_batched_kernel",
            "gelu_kernel",
            "gemm_batched_kernel",
            "add_kernel",
        ]
    counters, nodes = {}, []
    for (slot, inputs, output), target in zip(ATTENTION_GRAPH, targets, strict=True):
        ordinal = counters.get(target, 0)
        counters[target] = ordinal + 1
        if model == "qwen38" and group == 351 and slot == "residual2":
            ordinal = len(call_sites(owner, target)) - 1
        source = anchor(owner, target, ordinal)
        is_gemm = slot in {"qkv", "qk", "pv", "attention_projection", "mlp_fc", "mlp_projection"}
        method = (
            "custom_tiled_matmul"
            if is_gemm and model == "dsv4" and group == 349
            else (
                "custom_scalar_matmul"
                if model == "dsv4" and slot in {"qk", "pv"}
                else (
                    "cublas_matmul"
                    if is_gemm
                    else (
                        "cuda_layernorm"
                        if slot.startswith("ln")
                        else (
                            "cuda_pointwise_sum"
                            if slot.startswith("residual")
                            else (
                                "cuda_tanh_gelu"
                                if slot == "gelu"
                                else "cuda_softmax" if slot == "causal_softmax" else "cuda_layout_conversion"
                            )
                        )
                    )
                )
            )
        )
        nodes.append(
            node(
                slot,
                method,
                inputs,
                output,
                [source],
                index_contract={"status": "unknown", "reason": "no_local_numerical_validation"},
            )
        )
        if slot in {"qkv", "attention_projection", "mlp_fc", "mlp_projection"}:
            if model == "qwen38" and group == 348:
                helper = body(turn, target)
                nodes[-1]["bias_evidence"] = anchor(helper, "bias_add_kernel")
            else:
                following = [c for c in call_sites(owner, "bias_add_kernel") if c["line"] > source["line"]]
                if not following:
                    raise ValueError(f"missing bias epilogue source at {slot}")
                nodes[-1]["bias_evidence"] = evidence(owner, following[0])
                output_index = 12 if target == "cublasSgemm" else 2
                main_output = source["source"]["arguments"][output_index]
                bias_input = following[0]["arguments"][0]
                nodes[-1]["bias_contract"] = {
                    "status": "same_buffer_expression" if main_output == bias_input else "unresolved_buffer_relation",
                    "main_output": main_output,
                    "bias_input": bias_input,
                }
    mapped = {n["slot"]: n for n in nodes}
    # Reviewed fusion placement: the reference bundle can span QK and softmax.
    # Do not label every softmax kernel as if it also applies scale and mask.
    if model == "qwen38" and group == 351:
        placement = {"scale": "qk_cublas_alpha", "mask": "softmax_kernel_causal_predicate"}
    elif model == "dsv4" and group == 348:
        placement = {"scale": "qk_kernel_scale_argument", "mask": "qk_kernel_reference_mask_tensor"}
    else:
        placement = {"scale": "softmax_kernel_scale_argument", "mask": "softmax_kernel_causal_predicate"}
    for slot in ("qk", "causal_softmax"):
        mapped[slot]["normalization_placement"] = placement | {"evidence_scope": "human_reviewed_QK_and_softmax_path"}
    dims = dict(B=128, T=512, C=768, n_head=8, nh=8, hs=96, M=65536, batch=1024, C3=2304, C4=3072)
    if model == "qwen38" and group == 348:
        for slot, helper, op, roles, shape in [
            ("qk", "launch_batched_qk", "qk", {"Q": "Q", "K": "K", "S": "P"}, (512, 512, 96)),
            ("pv", "launch_batched_sv", "pv", {"S": "P", "V": "V", "O": "O"}, (512, 96, 512)),
        ]:
            driver_call = call_sites(owner, helper)[0]
            actual = bind_scalar_arguments(owner, driver_call, {"batch": 3, "T": 4, "hs": 5}, dims)
            mapped[slot].update(checked_gemm(body(turn, helper), 0, actual, roles, op, shape, True))
            mapped[slot]["driver_dimension_evidence"] = evidence(owner, driver_call)
        helper = body(turn, "launch_gemm_bias")
        for ordinal, (slot, N, K) in enumerate(
            [
                ("qkv", 2304, 768),
                ("attention_projection", 768, 768),
                ("mlp_fc", 3072, 768),
                ("mlp_projection", 768, 3072),
            ]
        ):
            driver_call = call_sites(owner, "launch_gemm_bias")[ordinal]
            actual = bind_scalar_arguments(owner, driver_call, {"M": 4, "Nf": 5, "K": 6}, dims)
            mapped[slot].update(
                checked_gemm(helper, 0, actual, {"A": "X", "W": "W", "Y": "Y"}, "linear", (65536, N, K))
            )
            mapped[slot]["driver_dimension_evidence"] = evidence(owner, driver_call)
    elif model == "qwen38":
        for slot, ordinal, op, roles, shape in [
            ("qk", 0, "qk", {"q_buf": "Q", "k_buf": "K", "att_buf": "P"}, (512, 512, 96)),
            ("pv", 1, "pv", {"v_buf": "V", "att_buf": "P", "q_buf": "O"}, (512, 96, 512)),
        ]:
            mapped[slot].update(checked_gemm(owner, ordinal, dims, roles, op, shape, True))
        for ordinal, (slot, N, K, X, W, Y) in enumerate(
            [
                ("qkv", 2304, 768, "ln1_out", "attn_qkv_w", "qkv_buf"),
                ("attention_projection", 768, 768, "y_reshaped", "attn_proj_w", "attn_proj_out"),
                ("mlp_fc", 3072, 768, "ln2_out", "mlp_fc_w", "mlp_hidden"),
                ("mlp_projection", 768, 3072, "mlp_hidden", "mlp_proj_w", "ln2_out"),
            ]
        ):
            mapped[slot].update(checked_gemm(owner, ordinal, dims, {X: "X", W: "W", Y: "Y"}, "linear", (65536, N, K)))
        mapped["residual1"]["storage_contract"] = {
            "status": (
                "uninitialized_read_before_later_reset" if turn["turn_idx"] == 1 else "direct_write_from_two_inputs"
            ),
            "evidence": evidence(owner),
        }
        mapped["pv"]["storage_contract"] = {
            "status": "reuses_Q_after_QK_consumer",
            "buffer_role": "q_heads_then_head_output",
        }
    elif group == 348:
        for ordinal, (slot, N, K, X, W, Y) in enumerate(
            [
                ("qkv", 2304, 768, "ln1_out", "c_attn_w", "qkv_buf"),
                ("attention_projection", 768, 768, "proj_buf", "c_proj_w", "temp_buf"),
                ("mlp_fc", 3072, 768, "ln1_out", "c_fc_w", "h_buf"),
                ("mlp_projection", 768, 3072, "h_buf", "mlp_cproj_w", "proj_buf"),
            ]
        ):
            mapped[slot].update(checked_gemm(owner, ordinal, dims, {X: "X", W: "W", Y: "Y"}, "linear", (65536, N, K)))
        qk = body(turn, "qk_matmul_kernel")
        expressions = re.findall(r"mask\[([^\]]+)\]", qk["source"])
        mapped["qk"]["mask_storage_contract"] = (
            check_mask_stride(expressions[0], dims | {"max_seqlen": 1024}, 1024)
            if len(expressions) == 1
            else {"status": "unknown", "reason": "mask_expression_unresolved"}
        ) | {"evidence": evidence(qk)}
    else:
        c = call_sites(owner, "gemm_batched_kernel")[3]
        args = c["arguments"]
        mapped["attention_projection"]["storage_contract"] = {
            "status": "input_output_alias_hazard" if args[0] == args[2] else "separate_input_output",
            "actual_input": args[0],
            "actual_output": args[2],
            "evidence": [evidence(owner, c), evidence(body(turn, "gemm_batched_kernel"))],
            "scope": "static_hazard_not_attribution_of_observed_runtime_failure",
        }
    return nodes
