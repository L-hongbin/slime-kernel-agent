"""Train the native Qwen MTP head with one-step teacher forcing.

Rollout can reuse this head for multiple speculative steps. Training predicts
only the immediate draft target from teacher inputs. Target features, token
embeddings and the vocabulary projection are stop-gradient inputs to this
auxiliary objective; the actor keeps its usual policy gradient.
"""

from copy import copy

import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler, MTPLossLoggingHelper, roll_tensor


class QwenMTPGPTModel(GPTModel):
    def __init__(self, **kwargs):
        config = kwargs["config"]
        if config.mtp_num_layers != 1:
            raise ValueError("Qwen MTP training requires exactly one native MTP layer")
        if config.virtual_pipeline_model_parallel_size or config.pipeline_model_parallel_layout is not None:
            raise ValueError("Qwen MTP training requires MTP on the final physical pipeline stage")
        super().__init__(**kwargs)
        if self.mtp_process:
            # The decoder uses block recompute. The native MTP layer needs
            # its own checkpoint; MCore otherwise skips block recompute.
            layer = self.mtp.layers[0]
            layer.config = copy(layer.config)
            if layer.config.recompute_granularity == "full":
                layer.config.recompute_method = "uniform"
                layer.config.recompute_num_layers = 1

    def _postprocess(self, *, hidden_states, input_ids, position_ids, labels, mtp_kwargs=None, **kwargs):
        mtp_labels = (mtp_kwargs or {}).get("mtp_labels")
        if self.mtp_process and mtp_labels is not None:
            if not self.post_process:
                raise RuntimeError("The native MTP head must be on the output pipeline stage")
            hidden_states = self._mtp_loss(hidden_states, input_ids, position_ids, mtp_labels, **kwargs)
        # MTP is already attached to the actor output's backward pass. Preserve
        # the ordinary GPT vocabulary projection and policy loss unchanged.
        return super()._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            mtp_kwargs=None,
            **kwargs,
        )

    def _mtp_loss(self, target_hidden, input_ids, position_ids, labels, **kwargs):
        packed = kwargs.get("packed_seq_params")

        def shift(tensor):
            return roll_tensor(tensor, shifts=-1, dims=-1, cp_group=self.cp_group, packed_seq_params=packed)[0]

        # Labels are unshifted supervision tokens. Trajectory packing may
        # restore a turn's terminal target without changing input_ids: later
        # turns must retain their actual (possibly template-repaired) history.
        # Depth 1 predicts x[t+2] from target h[t] and embedding x[t+1].
        labels = shift(labels.clone())
        mask = kwargs.get("loss_mask")
        # get_batch.full_loss_masks already selects the next-token target at
        # each hidden-state position (prompt_length - 1 left padding). Shift
        # that mask once, not once again with raw input IDs.
        if mask is None:
            mask = shift(torch.ones_like(labels))
        hidden = target_hidden.detach().requires_grad_(True)
        weight = (
            self.shared_embedding_or_output_weight()
            if self.share_embeddings_and_output_weights
            else self.output_layer.weight
        )
        weight = weight.detach()
        layer_kwargs = {
            name: kwargs.get(name)
            for name in (
                "attention_mask",
                "inference_params",
                "rotary_pos_emb",
                "rotary_pos_cos",
                "rotary_pos_sin",
                "packed_seq_params",
                "sequence_len_offset",
            )
        }
        layer_kwargs.update(kwargs.get("extra_block_kwargs") or {})
        loss_scale = self.config.mtp_loss_scaling_factor
        hidden, input_ids, position_ids = self.mtp.layers[0](
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden,
            embedding=self.embedding,
            **layer_kwargs,
        )
        labels = shift(labels)
        mask = shift(mask)

        def cross_entropy(features, targets):
            logits, _ = self.output_layer(
                input_=features,
                weight=weight,
                runtime_gather_output=kwargs.get("runtime_gather_output"),
            )
            return self.compute_language_model_loss(targets, logits)

        # Retain the small per-token losses instead of the vocabulary
        # tensor. Recompute uses the same frozen projection in backward.
        loss = tensor_parallel.checkpoint(cross_entropy, False, hidden, labels)
        loss = mask * loss
        # Normalize over the whole sequence, including CP ranks with no
        # valid response tokens. Averaging rank-local means would weight
        # short/empty CP slices differently.
        totals = torch.stack((loss.detach().sum(), mask.detach().sum()))
        cp_size = torch.distributed.get_world_size(self.cp_group)
        if cp_size > 1:
            torch.distributed.all_reduce(totals, group=self.cp_group)
        count = totals[1].clamp_min(1.0)
        if self.training:
            MTPLossLoggingHelper.save_loss_to_tracker(
                totals[0] / count,
                0,
                1,
                avg_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
            )
        # In per-token mode slime reports the full response-token count on
        # every CP rank. MCore sums both gradients and these duplicate counts,
        # so match the policy loss's CP compensation before finalization.
        auxiliary_loss = loss_scale * loss * cp_size
        if not self.config.calculate_per_token_loss:
            # Megatron averages parameter gradients over DP including CP.
            auxiliary_loss = auxiliary_loss / count
        target_hidden = MTPLossAutoScaler.apply(target_hidden, auxiliary_loss)
        return target_hidden
