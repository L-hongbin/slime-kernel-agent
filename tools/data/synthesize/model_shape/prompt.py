"""Versioned prompt and deterministic byte-granularity target sampler."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from tools.data.synthesize.augment_prompt_tasks import analyze_code
from tools.data.synthesize.shape_contract import (
    LARGE_INPUT_MAX_BYTES,
    MEDIUM_INPUT_MAX_BYTES,
    MEDIUM_INPUT_MIN_BYTES,
    MIN_INPUT_SCALE,
)

PROMPT_VERSION_V1 = "dsv4_shape_hardtail_user_only_v1"
PROMPT_VERSION = "dsv4_shape_hardtail_user_only_v2"
TARGET_SALT = "dsv4_shape_hardtail_byte_targets_v1"
VARIANTS = ("medium", "large")

USER_TEMPLATE_V1 = """Modify only the input shapes in this PyTorch reference and return two complete variants.

Hard constraints, in priority order:
1. Preserve the Model class, forward computation, imports, statements, tensor factories, dtypes, control flow, and tensor ranks exactly. Change only existing positive integer input-shape values in get_inputs(), plus matching numeric values in get_init_inputs() when the same logical dimension must stay coupled. Never compensate by rewriting code.
2. Keep every operator's shape relationships valid. Change the smallest coherent set of logical dimensions needed; this is usually 2-5 values, but use fewer or more when the program's actual constraints require it. Increase changed dimensions; do not create unused/dead tails.
3. Total storage of tensors returned by get_inputs() must be at least 2x the original and within 25% of the exact byte target while remaining in the named band:
   - Medium: exact target {medium_target_bytes} bytes ({medium_target_mib:.6f} MiB); band 64-256 MiB inclusive.
   - Large: exact target {large_target_bytes} bytes ({large_target_mib:.6f} MiB); band above 256 MiB through 4096 MiB inclusive.
4. Do not round or snap shape values to familiar anchors such as powers of two or multiples of 32. Use varied positive integers near the target. A power of two is allowed only when an existing operator constraint genuinely needs it; there is no required power-of-two quota.
5. In every returned input tensor with rank at least 2, the largest dimension must be no more than 1000 times its second-largest dimension.

If an exact target is infeasible, preserve correctness and choose the closest valid shape within the tolerance. Output exactly these Markdown sections, with no JSON and no prose outside them:

## Medium
```python
<complete reference>
```

## Large
```python
<complete reference>
```

PyTorch reference:
```python
{reference}
```
"""

USER_TEMPLATE = """Modify only the input shapes in this PyTorch reference and return two complete variants.

Hard constraints, in priority order:
1. Preserve the Model class, forward computation, imports, statements, tensor factories, dtypes, control flow, and tensor ranks exactly. Change only existing positive integer input-shape values in get_inputs(), plus matching numeric values in get_init_inputs() when the same logical dimension must stay coupled. Never compensate by rewriting code.
2. Keep every operator's shape relationships valid and do not create unused/dead tails. Change the smallest coherent set of logical dimensions needed; this is usually 2-5 values, but use fewer or more when the program requires it. Before returning, compare each variant with the source: every edited integer must contribute to a returned input tensor's shape or be its necessary coupled init value, and every edited value must be larger than the original.
3. Total storage of tensors returned by get_inputs() must be at least 2x the original and within 25% of the exact byte target while remaining in the named band:
   - Medium: exact target {medium_target_bytes} bytes ({medium_target_mib:.6f} MiB); band 64-256 MiB inclusive.
   - Large: exact target {large_target_bytes} bytes ({large_target_mib:.6f} MiB); band above 256 MiB through 4096 MiB inclusive.
   The tolerance is deliberate: once a valid in-band shape is within 25%, stop refining arithmetic instead of chasing an exact factorization.
4. Do not enforce a power-of-two quota and do not systematically prefer or avoid powers of two, round decimal values, or values just beside familiar anchors. Choose varied positive integers from the actual operator constraints and target.
5. In every returned input tensor with rank at least 2, the largest dimension must be no more than 1000 times its second-largest dimension.

If an exact target is infeasible, preserve correctness and choose the closest valid shape within the tolerance. Output exactly these Markdown sections, with no JSON and no prose outside them:

## Medium
```python
<complete reference>
```

## Large
```python
<complete reference>
```

PyTorch reference:
```python
{reference}
```
"""

USER_TEMPLATES = {
    PROMPT_VERSION_V1: USER_TEMPLATE_V1,
    PROMPT_VERSION: USER_TEMPLATE,
}


def _nested(value: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _sample_bytes(identity: str, lower: int, upper: int) -> int:
    if lower > upper:
        raise ValueError(f"target_band_infeasible:{lower}:{upper}")
    digest = hashlib.sha256(f"{TARGET_SALT}:{identity}".encode()).digest()
    target = lower + int.from_bytes(digest[:8], "big") % (upper - lower + 1)
    # A uniformly sampled exact byte count almost never lands on these anchors;
    # make the contract explicit and deterministic instead of relying on chance.
    if target % (1024**2) == 0 or target & (target - 1) == 0:
        target = target + 1 if target < upper else target - 1
    return target


def target_input_bytes(row: Mapping[str, Any]) -> dict[str, int]:
    uuid = _nested(row, "extra_info.uuid")
    reference = _nested(row, "reward_model.ground_truth")
    entry_point = str(_nested(row, "extra_info.entry_point", "Model"))
    if not isinstance(uuid, str) or not isinstance(reference, str):
        raise ValueError("parent_requires_uuid_and_reference")
    parent_bytes = int(analyze_code(reference, entry_point).input_bytes)
    minimum = max(parent_bytes * 2, int(parent_bytes * MIN_INPUT_SCALE))
    medium_lower = max(MEDIUM_INPUT_MIN_BYTES, minimum)
    large_lower = max(MEDIUM_INPUT_MAX_BYTES + 1, minimum)
    return {
        "medium": _sample_bytes(f"{uuid}:medium", medium_lower, MEDIUM_INPUT_MAX_BYTES),
        "large": _sample_bytes(f"{uuid}:large", large_lower, LARGE_INPUT_MAX_BYTES),
    }


def render_user_prompt(
    row: Mapping[str, Any],
    targets: Mapping[str, int],
    prompt_version: str = PROMPT_VERSION,
) -> str:
    reference = _nested(row, "reward_model.ground_truth")
    if not isinstance(reference, str):
        raise ValueError("parent_reference_missing")
    medium = int(targets["medium"])
    large = int(targets["large"])
    try:
        template = USER_TEMPLATES[prompt_version]
    except KeyError:
        raise ValueError(f"unsupported_prompt_version:{prompt_version}") from None
    return template.format(
        reference=reference.rstrip(),
        medium_target_bytes=medium,
        medium_target_mib=medium / 1024**2,
        large_target_bytes=large,
        large_target_mib=large / 1024**2,
    )
