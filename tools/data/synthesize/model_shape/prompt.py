"""Versioned prompt and deterministic byte-granularity target sampler."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from tools.data.synthesize.augment_prompt_tasks import analyze_code
from tools.data.synthesize.model_shape.shape_contract import (
    LARGE_INPUT_MAX_BYTES,
    MEDIUM_INPUT_MAX_BYTES,
    MEDIUM_INPUT_MIN_BYTES,
    MIN_INPUT_SCALE,
)

PROMPT_VERSION_V1 = "dsv4_shape_hardtail_user_only_v1"
PROMPT_VERSION = "dsv4_shape_hardtail_user_only_v2"
PROMPT_VERSION_RETRY_V1 = "dsv4_shape_retry_user_only_v1"
PROMPT_VERSION_RETRY_V2 = "dsv4_shape_retry_user_only_v2"
PROMPT_VERSION_RETRY = "dsv4_shape_retry_user_only_v3"
PROMPT_VERSION_UNPROFILED_V1 = "dsv4_shape_unprofiled_user_only_v1"
PROMPT_VERSION_UNPROFILED_V2 = "dsv4_shape_unprofiled_user_only_v2"
PROMPT_VERSION_UNPROFILED = "dsv4_shape_unprofiled_user_only_v3"
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

USER_TEMPLATE_RETRY_V1 = """A previous shape-only proposal for this PyTorch reference did not pass validation. Solve it again from the source and return two complete variants.

Hard constraints, in priority order:
1. Preserve the complete program exactly except for positive integer input-shape values in get_inputs(), and matching integer values in get_init_inputs() only when the same logical dimension must remain coupled. Do not edit imports, comments, formatting, names, operations, statements, dtypes, control flow, ranks, or the forward computation.
2. Trace the actual operator constraints before choosing values. Every changed integer must affect a returned input tensor shape or be a necessary coupled init value. Increase changed dimensions, preserve every required equality/divisibility/product relation, and avoid unused/dead tails or oversized intermediates.
3. Total storage of tensors returned by get_inputs() must be at least 2x the original and within 25% of the target while remaining in the named band:
   - Medium: target {medium_target_bytes} bytes ({medium_target_mib:.6f} MiB); band 64-256 MiB inclusive.
   - Large: target {large_target_bytes} bytes ({large_target_mib:.6f} MiB); band above 256 MiB through 4096 MiB inclusive.
4. Use however many coupled shape values correctness requires. Do not impose a fixed slot count or a power-of-two pattern, and do not systematically snap values to familiar binary or decimal anchors.
5. In every returned input tensor with rank at least 2, the largest dimension must be no more than 1000 times its second-largest dimension.

If the exact target is infeasible, choose the closest safe in-band shape within the tolerance. Check each full reference against the unchanged source before returning it. Output exactly these Markdown sections, with no JSON and no prose outside them:

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

USER_TEMPLATE_RETRY_V2 = """A previous shape-only proposal for this PyTorch reference did not pass validation. Solve it again from the source and return two complete variants.

Hard constraints, in priority order:
1. Preserve the complete program exactly except for positive integer input-shape values in get_inputs(), and matching integer values in get_init_inputs() only when the same logical dimension must remain coupled. Do not edit imports, comments, formatting, names, operations, statements, dtypes, control flow, ranks, or the forward computation.
2. Trace the actual operator constraints before choosing values. Every changed integer must affect a returned input tensor shape or be a necessary coupled init value. Increase changed dimensions, preserve every required equality/divisibility/product relation, and do not create unused/dead tails.
3. Keep each complete program realistically runnable on one H20. Estimate parameter, intermediate, temporary, and final-output shapes, especially when an operation grows faster than its input. Runtime correctness and feasible peak storage take priority over target proximity: when a target would make any materialized tensor or peak live storage impractical, choose a smaller safe increased shape instead.
4. Subject to runtime feasibility, total storage of tensors returned by get_inputs() should be within 25% of the target and in the named band:
   - Medium: target {medium_target_bytes} bytes ({medium_target_mib:.6f} MiB); band 64-256 MiB inclusive.
   - Large: target {large_target_bytes} bytes ({large_target_mib:.6f} MiB); band above 256 MiB through 4096 MiB inclusive.
   If no safe shape exists in the requested band, return the closest safe increased shape rather than an un-runnable target-sized input.
5. Use however many coupled shape values correctness requires. Do not impose a fixed slot count or a power-of-two pattern, and do not systematically snap values to familiar binary or decimal anchors.
6. In every returned input tensor with rank at least 2, the largest dimension must be no more than 1000 times its second-largest dimension.

Check shape coupling and estimated peak storage for each full reference before returning it. Output exactly these Markdown sections, with no JSON and no prose outside them:

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

USER_TEMPLATE_RETRY = USER_TEMPLATE_RETRY_V2.replace(
    "Do not impose a fixed slot count or a power-of-two pattern, and do not systematically snap values to familiar binary or decimal anchors.",
    "Do not impose a fixed slot count and do not optimize for or against binary or decimal anchors. Never perturb a naturally valid anchored value by a small amount merely to appear irregular; choose anchored or irregular values only from the operator constraints and storage target.",
)

USER_TEMPLATE_UNPROFILED_V1 = """Modify only the input shapes in this PyTorch reference and return two complete variants. Automatic tooling could not establish this source's input profile, so infer its tensor factories, dtypes, shape coupling, and operator constraints directly from the code.

Hard constraints, in priority order:
1. Preserve the complete program exactly except for positive integer input-shape values in get_inputs(), and matching integer values in get_init_inputs() only when the same logical dimension must remain coupled. Do not edit imports, comments, formatting, names, operations, statements, dtypes, control flow, ranks, or the forward computation.
2. Every changed integer must affect a returned input tensor shape or be a necessary coupled init value. Increase changed dimensions, preserve every required equality/divisibility/product relation, and avoid unused/dead tails or oversized intermediates.
3. Infer aggregate input storage from the source and aim within 25% of these deliberately irregular targets:
   - Medium: target {medium_target_bytes} bytes ({medium_target_mib:.6f} MiB); band 64-256 MiB inclusive.
   - Large: target {large_target_bytes} bytes ({large_target_mib:.6f} MiB); band above 256 MiB through 4096 MiB inclusive.
   Prefer correctness and an increased input over exact arithmetic when the automatic-profile failure makes a target infeasible.
4. Use however many coupled shape values correctness requires. Do not impose a fixed slot count or a power-of-two pattern, and do not systematically snap values to familiar binary or decimal anchors.
5. In every returned input tensor with rank at least 2, the largest dimension must be no more than 1000 times its second-largest dimension.

Check each full reference against the unchanged source before returning it. Output exactly these Markdown sections, with no JSON and no prose outside them:

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

USER_TEMPLATE_UNPROFILED_V2 = (
    USER_TEMPLATE_UNPROFILED_V1.replace(
        "Every changed integer must affect a returned input tensor shape or be a necessary coupled init value. Increase changed dimensions, preserve every required equality/divisibility/product relation, and avoid unused/dead tails or oversized intermediates.",
        "Every changed integer must affect a returned input tensor shape or be a necessary coupled init value. Increase changed dimensions, preserve every required equality/divisibility/product relation, and avoid unused/dead tails. Keep each complete program realistically runnable on one H20: estimate parameter, intermediate, temporary, and final-output shapes, and prioritize feasible peak storage over target proximity.",
    )
    .replace(
        "Prefer correctness and an increased input over exact arithmetic when the automatic-profile failure makes a target infeasible.",
        "Prefer correctness, feasible peak storage, and an increased input over exact arithmetic. If no safe shape exists in a requested band, return the closest safe increased shape rather than an un-runnable target-sized input.",
    )
    .replace(
        "Do not impose a fixed slot count or a power-of-two pattern, and do not systematically snap values to familiar binary or decimal anchors.",
        "Do not impose a fixed slot count and do not optimize for or against binary or decimal anchors. Never perturb a naturally valid anchored value by a small amount merely to appear irregular; choose anchored or irregular values only from the operator constraints and storage target.",
    )
)

USER_TEMPLATE_UNPROFILED = (
    USER_TEMPLATE_UNPROFILED_V2.replace(
        "Do not edit imports, comments, formatting, names, operations, statements, dtypes, control flow, ranks, or the forward computation.",
        "Do not edit imports, comments, formatting, names, operations, statements, dtypes, control flow, ranks, or the forward computation. At every shape site preserve the existing Name, Subscript, BinOp, Tuple, and call structure exactly: replace only an existing integer token in place. If a shape uses a name or expression, edit its defining integer literal; never inline, expand, simplify, or constant-fold that name or expression.",
    )
    .replace(
        "Every changed integer must affect a returned input tensor shape or be a necessary coupled init value.",
        "Every changed integer must affect a returned input tensor shape or be a necessary coupled init value. Leave dead, redundant, or merely shape-looking assignments unchanged.",
    )
    .replace(
        "Check each full reference against the unchanged source before returning it.",
        "Do not enumerate or repeat factor searches; once safe values are found, output immediately. Check each full reference against the unchanged source before returning it.",
    )
)

USER_TEMPLATES = {
    PROMPT_VERSION_V1: USER_TEMPLATE_V1,
    PROMPT_VERSION: USER_TEMPLATE,
    PROMPT_VERSION_RETRY_V1: USER_TEMPLATE_RETRY_V1,
    PROMPT_VERSION_RETRY_V2: USER_TEMPLATE_RETRY_V2,
    PROMPT_VERSION_RETRY: USER_TEMPLATE_RETRY,
    PROMPT_VERSION_UNPROFILED_V1: USER_TEMPLATE_UNPROFILED_V1,
    PROMPT_VERSION_UNPROFILED_V2: USER_TEMPLATE_UNPROFILED_V2,
    PROMPT_VERSION_UNPROFILED: USER_TEMPLATE_UNPROFILED,
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


def target_input_bytes_from_size(uuid: str, parent_bytes: int) -> dict[str, int]:
    if not uuid or parent_bytes <= 0:
        raise ValueError("target_sampling_requires_uuid_and_positive_parent_bytes")
    minimum = max(parent_bytes * 2, int(parent_bytes * MIN_INPUT_SCALE))
    medium_lower = max(MEDIUM_INPUT_MIN_BYTES, minimum)
    large_lower = max(MEDIUM_INPUT_MAX_BYTES + 1, minimum)
    return {
        "medium": _sample_bytes(f"{uuid}:medium", medium_lower, MEDIUM_INPUT_MAX_BYTES),
        "large": _sample_bytes(f"{uuid}:large", large_lower, LARGE_INPUT_MAX_BYTES),
    }


def target_input_bytes_without_profile(uuid: str) -> dict[str, int]:
    """Sample both target bands when source storage cannot be established."""

    if not uuid:
        raise ValueError("target_sampling_requires_uuid")
    return {
        "medium": _sample_bytes(f"unprofiled:{uuid}:medium", MEDIUM_INPUT_MIN_BYTES, MEDIUM_INPUT_MAX_BYTES),
        "large": _sample_bytes(
            f"unprofiled:{uuid}:large",
            MEDIUM_INPUT_MAX_BYTES + 1,
            LARGE_INPUT_MAX_BYTES,
        ),
    }


def target_input_bytes(row: Mapping[str, Any]) -> dict[str, int]:
    uuid = _nested(row, "extra_info.uuid")
    reference = _nested(row, "reward_model.ground_truth")
    entry_point = str(_nested(row, "extra_info.entry_point", "Model"))
    if not isinstance(uuid, str) or not isinstance(reference, str):
        raise ValueError("parent_requires_uuid_and_reference")
    parent_bytes = int(analyze_code(reference, entry_point).input_bytes)
    return target_input_bytes_from_size(uuid, parent_bytes)


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
