"""Small helpers for bounding DrKernel eval coroutine fan-out."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from typing import Any


def get_positive_int_env(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return 0
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


async def run_eval_coro(coro_factory: Callable[[], Any], semaphore: asyncio.Semaphore | None) -> Any:
    if semaphore is None:
        return await coro_factory()
    async with semaphore:
        return await coro_factory()
