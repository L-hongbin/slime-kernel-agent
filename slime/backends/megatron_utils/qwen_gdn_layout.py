from __future__ import annotations

import re

import torch

_GDN_FACTORY_SECTIONS = {
    "self_attention.in_proj.weight": ("query", "key", "value", "z", "beta", "alpha"),
    "self_attention.conv1d.weight": ("query", "key", "value"),
}


def interleave_gdn_tp_sections(sections: list[torch.Tensor], tp_size: int, dim: int = 0) -> torch.Tensor:
    """Pack global GDN sections into the rank-major layout of a native fused parameter."""
    if not sections:
        raise ValueError("GDN section list must not be empty.")
    if tp_size < 1:
        raise ValueError(f"tp_size must be positive, got {tp_size}.")
    if any(section.shape[dim] % tp_size != 0 for section in sections):
        raise ValueError(
            f"Every GDN section must be divisible by tp_size={tp_size}, "
            f"got {[section.shape[dim] for section in sections]} along dim={dim}."
        )

    per_section_shards = [section.chunk(tp_size, dim=dim) for section in sections]
    rank_shards = [
        torch.cat([section_shards[rank] for section_shards in per_section_shards], dim=dim) for rank in range(tp_size)
    ]
    return torch.cat(rank_shards, dim=dim)


def get_gdn_factory_group(key: str) -> tuple[str, tuple[str, ...]] | None:
    """Return the fused parameter key and ordered factory sections for a raw DCP key."""
    for parameter_suffix, section_names in _GDN_FACTORY_SECTIONS.items():
        for section_name in section_names:
            section_suffix = f"{parameter_suffix}.{section_name}"
            if key.endswith(section_suffix):
                return key[: -len(section_name) - 1], section_names
    return None


def group_gdn_factory_keys(keys: list[str]) -> list[list[str]]:
    """Group raw DCP factory keys that must be loaded by the same conversion worker."""
    factory_groups: dict[str, tuple[tuple[str, ...], dict[str, str]]] = {}
    for key in keys:
        group = get_gdn_factory_group(key)
        if group is None:
            continue
        fused_key, section_names = group
        _, section_keys = factory_groups.setdefault(fused_key, (section_names, {}))
        section_keys[key.rsplit(".", 1)[-1]] = key

    key_groups = []
    for fused_key, (section_names, section_keys) in factory_groups.items():
        missing_sections = [name for name in section_names if name not in section_keys]
        if missing_sections:
            raise ValueError(f"Missing GDN factory sections for {fused_key}: {missing_sections}.")
        key_groups.append([section_keys[name] for name in section_names])
    return key_groups


def merge_gdn_factory_tensors(state_dict: dict[str, torch.Tensor], tp_size: int, implementation: str) -> list[str]:
    """Restore native fused GDN tensors from raw ShardedTensorFactory checkpoint entries."""
    if implementation not in {"auto", "replicated", "distributed"}:
        raise ValueError(f"Unsupported Qwen GDN implementation: {implementation}.")

    factory_groups: dict[str, tuple[tuple[str, ...], dict[str, str]]] = {}
    for key in state_dict:
        group = get_gdn_factory_group(key)
        if group is None:
            continue
        fused_key, section_names = group
        _, section_keys = factory_groups.setdefault(fused_key, (section_names, {}))
        section_keys[key.rsplit(".", 1)[-1]] = key

    if not factory_groups:
        return []
    if implementation == "replicated":
        raise ValueError(
            "The checkpoint contains distributed Qwen GDN factory tensors. "
            "Use --qwen-gdn-implementation distributed or auto."
        )

    merged_keys = []
    for fused_key, (section_names, section_keys) in factory_groups.items():
        missing_sections = [name for name in section_names if name not in section_keys]
        if missing_sections:
            raise ValueError(f"Missing GDN factory sections for {fused_key}: {missing_sections}.")
        if fused_key in state_dict:
            raise ValueError(f"Both fused and factory-split GDN tensors exist for {fused_key}.")

        # Raw DCP metadata omits the ShardedTensorFactory merge function. Explicit
        # layer keys store the TP-sharded dimension at axis 0, while layer-stacked
        # checkpoints add the layer dimension at axis 0 and move it to axis 1.
        if re.search(r"\.layers\.\d+\.", fused_key):
            section_dim = 0
        elif ".layers." in fused_key:
            section_dim = 1
        else:
            section_dim = 0
        state_dict[fused_key] = interleave_gdn_tp_sections(
            [state_dict[section_keys[name]] for name in section_names], tp_size, dim=section_dim
        )
        for key in section_keys.values():
            del state_dict[key]
        merged_keys.append(fused_key)

    return merged_keys


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
