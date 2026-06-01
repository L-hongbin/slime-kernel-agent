from __future__ import annotations

import asyncio

import pytest

from slime_plugins.drkernel.eval_throttle import get_positive_int_env, run_eval_coro


@pytest.mark.unit
def test_eval_coro_throttle_caps_concurrency():
    async def run() -> int:
        semaphore = asyncio.Semaphore(2)
        active = 0
        max_active = 0
        lock = asyncio.Lock()

        async def factory():
            nonlocal active, max_active
            async with lock:
                active += 1
                max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            async with lock:
                active -= 1
            return True

        await asyncio.gather(*(run_eval_coro(factory, semaphore) for _ in range(5)))
        return max_active

    assert asyncio.run(run()) == 2


@pytest.mark.unit
def test_eval_concurrency_env_rejects_negative(monkeypatch):
    monkeypatch.delenv("DRKERNEL_EVAL_MAX_CONCURRENCY", raising=False)
    assert get_positive_int_env("DRKERNEL_EVAL_MAX_CONCURRENCY") == 0

    monkeypatch.setenv("DRKERNEL_EVAL_MAX_CONCURRENCY", "4")
    assert get_positive_int_env("DRKERNEL_EVAL_MAX_CONCURRENCY") == 4

    monkeypatch.setenv("DRKERNEL_EVAL_MAX_CONCURRENCY", "-1")
    with pytest.raises(ValueError, match="DRKERNEL_EVAL_MAX_CONCURRENCY"):
        get_positive_int_env("DRKERNEL_EVAL_MAX_CONCURRENCY")

    monkeypatch.setenv("DRKERNEL_EVAL_MAX_CONCURRENCY", "bad")
    with pytest.raises(ValueError, match="must be an integer"):
        get_positive_int_env("DRKERNEL_EVAL_MAX_CONCURRENCY")
