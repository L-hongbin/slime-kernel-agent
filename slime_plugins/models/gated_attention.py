"""Compatibility for gated attention with fewer KV heads than TP ranks."""

from megatron.core.transformer.attention import SelfAttention


class TPGatedSelfAttention(SelfAttention):
    """Backport Megatron #3575 without changing the installed Megatron source.

    Older versions slice query heads after gathering a shared KV group, but
    leave the corresponding gate unsliced. Newer versions already return a
    rank-local gate and pass through unchanged. No parameters or state-dict
    keys are added, and the slice retains the upstream autograd graph.
    """

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None, output_gate=False, split_qkv=True):
        qkv = super().get_query_key_value_tensors(
            hidden_states, key_value_states=key_value_states, output_gate=output_gate, split_qkv=split_qkv
        )
        if not output_gate:
            return qkv

        query, key, value, gate = qkv
        if gate.shape == query.shape:
            return qkv

        tp_size = self.world_size
        num_groups = self.config.num_query_groups
        num_heads = self.config.num_attention_heads
        if not (
            num_groups is not None
            and 0 < num_groups < tp_size
            and tp_size % num_groups == 0
            and num_heads % tp_size == 0
            and query.ndim == gate.ndim == 4
            and query.shape[:2] == gate.shape[:2]
            and query.shape[-1] == gate.shape[-1] == self.hidden_size_per_attention_head
            and query.shape[2] == num_heads // tp_size
            and gate.shape[2] == num_heads // num_groups
        ):
            raise RuntimeError(
                "Unexpected gated attention layout: "
                f"query={tuple(query.shape)}, gate={tuple(gate.shape)}, TP={tp_size}, "
                f"num_attention_heads={num_heads}, num_query_groups={num_groups}"
            )

        # Use this module's TP group, not the global TP rank (which can differ
        # with custom process-group collections).
        rank = self.pg_collection.tp.rank()
        if not 0 <= rank < tp_size:
            raise RuntimeError(f"Invalid gated attention TP group rank {rank} for TP={tp_size}")
        heads_per_rank = query.shape[2]
        start = (rank % (tp_size // num_groups)) * heads_per_rank
        gate = gate.narrow(2, start, heads_per_rank)
        return query, key, value, gate
