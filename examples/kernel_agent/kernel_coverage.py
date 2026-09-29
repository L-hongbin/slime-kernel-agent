"""Coverage metrics shared by reward and rejection sampling."""

import math
from typing import Any


def _efficiency_reference_coverage(result, metadata, custom_time, total_time):
    """Discount reference coverage only when the candidate is slower than reference.

    Profiler times are microseconds; both end-to-end runtimes are milliseconds.
    Invalid measurements earn no coverage credit, while the reward caller retains
    its existing correctness/performance/failed-result handling.
    """
    diagnostics = {
        "coverage_reference_fraction": 0.0,
        "coverage_efficiency_discount": 0.0,
        "coverage_invalid_reason": None,
    }
    values = {
        "custom profiling time": custom_time,
        "total profiling time": total_time,
        "reference runtime": result.get("reference_runtime", metadata.get("reference_runtime")),
        "candidate runtime": result.get("kernel_runtime", metadata.get("kernel_runtime")),
    }
    try:
        if metadata.get("coverage_measurement_valid") is False:
            raise ValueError("profiler marked coverage measurement invalid")
        for name, value in values.items():
            if isinstance(value, bool):
                raise ValueError(f"{name} must be numeric, not bool")
            values[name] = float(value)
            if not math.isfinite(values[name]):
                raise ValueError(f"{name} must be finite")
        custom = values["custom profiling time"]
        total = values["total profiling time"]
        reference = values["reference runtime"]
        candidate = values["candidate runtime"]
        if custom < 0 or total <= 0 or reference <= 0 or candidate <= 0:
            raise ValueError("profiling total and runtimes must be positive; custom time must be non-negative")
        if custom > total and not math.isclose(custom, total, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("custom profiling time exceeds total profiling time")
        # No measured custom work must not earn credit from profiler/reference
        # timing differences alone. Preserve original refcov for PRS separately.
        if custom == 0:
            return 0.0, diagnostics
        noncustom_us = max(total - custom, 0.0)
        reference_fraction = min(max(1.0 - noncustom_us / reference / 1000.0, 0.0), 1.0)
        discount = min(1.0, reference / candidate)
        diagnostics["coverage_reference_fraction"] = reference_fraction
        diagnostics["coverage_efficiency_discount"] = discount
        return reference_fraction * discount, diagnostics
    except (ValueError, TypeError, OverflowError) as exc:
        diagnostics["coverage_invalid_reason"] = str(exc)
        return 0.0, diagnostics


def _compute_coverage(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    num_custom_kernel = result.get("num_custom_kernel", metadata.get("num_custom_kernel", 0)) or 0
    num_total_kernels = result.get("num_total_kernels", metadata.get("num_total_kernels", 0)) or 0
    custom_time = (
        result.get(
            "custom_kernel_cuda_time_in_profiling_us",
            metadata.get("custom_kernel_cuda_time_in_profiling_us", 0),
        )
        or 0
    )
    total_time = (
        result.get(
            "total_kernel_run_time_in_profiling_us",
            metadata.get("total_kernel_run_time_in_profiling_us", 0),
        )
        or 0
    )

    if config["coverage_reward_type"] in {"efficiency_reference_time_coverage", "capped_speed_auxiliary", "gated_time_coverage"}:
        coverage, diagnostics = _efficiency_reference_coverage(result, metadata, custom_time, total_time)
        if config["coverage_reward_type"] == "capped_speed_auxiliary":
            # Speed-only auxiliary reward, with exactly the same measurement and
            # custom-work gates as Effrefcov. PRS still uses original ref coverage.
            coverage = diagnostics["coverage_efficiency_discount"]
        elif config["coverage_reward_type"] == "gated_time_coverage":
            # Reuse the exact current measurement/custom-work gate. Only the
            # eligible auxiliary score changes to the historical custom ratio.
            # A zero reference fraction is valid and must not suppress Timecov.
            coverage = (
                float(custom_time) / float(total_time)
                if diagnostics["coverage_efficiency_discount"] > 0.0
                else 0.0
            )
        return {
            "coverage": coverage,
            "num_custom_kernel": num_custom_kernel,
            "num_total_kernels": num_total_kernels,
            "custom_kernel_cuda_time_in_profiling_us": custom_time,
            "total_kernel_run_time_in_profiling_us": total_time,
            **diagnostics,
        }

    number_coverage = float(num_custom_kernel) / float(num_total_kernels) if num_total_kernels else 0.0
    time_coverage = float(custom_time) / float(total_time) if total_time else 0.0
    coverage_type = config["coverage_reward_type"]
    if coverage_type == "reference_time_coverage":
        # KernelGym reports reference_runtime in ms and profiler kernel sums in us.
        # Keep the reference fixed (reference cache), independent of custom speed.
        reference_ms = result.get("reference_runtime", metadata.get("reference_runtime"))
        if reference_ms is None or not math.isfinite(float(reference_ms)) or float(reference_ms) <= 0:
            raise ValueError("reference_time_coverage requires positive finite reference_runtime in ms")
        if not math.isfinite(float(total_time)) or float(total_time) <= 0:
            raise ValueError("reference_time_coverage requires positive finite total profiling time in us")
        if not math.isfinite(float(custom_time)) or float(custom_time) < 0:
            raise ValueError("reference_time_coverage requires non-negative finite custom profiling time in us")
        coverage = min(max(1.0 - (float(total_time) - float(custom_time)) / (float(reference_ms) * 1000.0), 0.0), 1.0)
    elif coverage_type == "time_coverage":
        coverage = time_coverage
    elif coverage_type == "number_coverage":
        coverage = number_coverage
    else:
        raise ValueError(f"Unknown coverage reward type: {coverage_type!r}")
    return {
        "coverage": coverage,
        "num_custom_kernel": num_custom_kernel,
        "num_total_kernels": num_total_kernels,
        "custom_kernel_cuda_time_in_profiling_us": custom_time,
        "total_kernel_run_time_in_profiling_us": total_time,
    }
