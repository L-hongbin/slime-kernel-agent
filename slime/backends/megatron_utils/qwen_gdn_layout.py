from __future__ import annotations

import torch


def interleave_gdn_tp_sections(sections: list[torch.Tensor], tp_size: int) -> torch.Tensor:
    """Pack global GDN sections into the rank-major layout of a native fused parameter."""
    if not sections:
        raise ValueError("GDN section list must not be empty.")
    if tp_size < 1:
        raise ValueError(f"tp_size must be positive, got {tp_size}.")
    if any(section.shape[0] % tp_size != 0 for section in sections):
        raise ValueError(
            f"Every GDN section must be divisible by tp_size={tp_size}, "
            f"got {[section.shape[0] for section in sections]}."
        )

    per_section_shards = [section.chunk(tp_size, dim=0) for section in sections]
    rank_shards = [
        torch.cat([section_shards[rank] for section_shards in per_section_shards], dim=0) for rank in range(tp_size)
    ]
    return torch.cat(rank_shards, dim=0)


def deinterleave_gdn_tp_sections(
    tensor: torch.Tensor, section_sizes: tuple[int, ...], tp_size: int
) -> list[torch.Tensor]:
    """Unpack a gathered rank-major native GDN parameter into global logical sections."""
    if any(section % tp_size != 0 for section in section_sizes):
        raise ValueError(f"Every GDN section must be divisible by tp_size={tp_size}, got {section_sizes}.")
    if tensor.shape[0] != sum(section_sizes):
        raise ValueError(f"GDN tensor first dimension {tensor.shape[0]} does not match sections {section_sizes}.")

    local_sizes = tuple(section // tp_size for section in section_sizes)
    rank_size = sum(local_sizes)
    rank_shards = tensor.split(rank_size, dim=0)
    if len(rank_shards) != tp_size:
        raise ValueError(f"Expected {tp_size} GDN rank shards, got {len(rank_shards)}.")
    split_rank_shards = [rank_shard.split(local_sizes, dim=0) for rank_shard in rank_shards]
    return [
        torch.cat([rank_sections[section_idx] for rank_sections in split_rank_shards], dim=0)
        for section_idx in range(len(section_sizes))
    ]
