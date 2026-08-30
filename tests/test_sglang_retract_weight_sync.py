from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.patch_sglang_retract_flush import _MARKER, patch_file
from slime.backends.megatron_utils.update_weight import common
from slime.backends.sglang_utils import sglang_engine
from slime.utils.arguments import get_slime_extra_args_provider, slime_validate_args

pytestmark = pytest.mark.unit
NUM_GPUS = 0


class _RemoteMethod:
    def __init__(self, name: str, calls: list[tuple[str, dict]]):
        self.name = name
        self.calls = calls

    def remote(self, **kwargs):
        self.calls.append((self.name, kwargs))
        return (self.name, kwargs)


class _Engine:
    def __init__(self, calls: list[tuple[str, dict]]):
        self.pause_generation = _RemoteMethod("pause", calls)
        self.flush_cache = _RemoteMethod("flush", calls)


def test_pause_and_flush_uses_retract_before_cache_invalidation(monkeypatch):
    calls: list[tuple[str, dict]] = []
    ray_get_calls = []
    monkeypatch.setattr(common.ray, "get", lambda refs: ray_get_calls.append(refs))

    common.pause_and_flush_rollout_engines(
        SimpleNamespace(rollout_weight_sync_pause_mode="retract"),
        [_Engine(calls), _Engine(calls)],
    )

    assert calls == [
        ("pause", {"mode": "retract"}),
        ("pause", {"mode": "retract"}),
        ("flush", {}),
        ("flush", {}),
    ]
    assert len(ray_get_calls) == 2


def test_weight_sync_guard_resumes_after_success(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(common.ray, "get", lambda refs: refs)
    engine = _Engine(calls)
    engine.continue_generation = _RemoteMethod("continue", calls)

    with common.rollout_engine_weight_sync(
        SimpleNamespace(rollout_weight_sync_pause_mode="retract"),
        [engine],
        enabled=True,
    ):
        calls.append(("refit", {}))

    assert calls == [
        ("pause", {"mode": "retract"}),
        ("flush", {}),
        ("refit", {}),
        ("continue", {}),
    ]


def test_weight_sync_guard_aborts_parked_requests_after_failure(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(common.ray, "get", lambda refs: refs)
    engine = _Engine(calls)
    engine.continue_generation = _RemoteMethod("continue", calls)

    with pytest.raises(RuntimeError, match="injected refit failure"):
        with common.rollout_engine_weight_sync(
            SimpleNamespace(rollout_weight_sync_pause_mode="retract"),
            [engine],
            enabled=True,
        ):
            raise RuntimeError("injected refit failure")

    assert calls == [
        ("pause", {"mode": "retract"}),
        ("flush", {}),
        ("pause", {"mode": "abort"}),
        ("continue", {}),
    ]


def test_sglang_engine_pause_generation_sends_explicit_mode(monkeypatch):
    calls = []

    class _Response:
        def raise_for_status(self):
            return None

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return _Response()

    monkeypatch.setattr(sglang_engine.requests, "post", fake_post)
    engine = sglang_engine.SGLangEngine.__new__(sglang_engine.SGLangEngine)
    engine.server_host = "127.0.0.1"
    engine.server_port = 30000

    engine.pause_generation(mode="retract")
    engine.pause_generation()

    assert calls == [
        ("http://127.0.0.1:30000/pause_generation", {"json": {"mode": "retract"}}),
        ("http://127.0.0.1:30000/pause_generation", {"json": {"mode": "abort"}}),
    ]
    with pytest.raises(ValueError, match="Unsupported SGLang pause mode"):
        engine.pause_generation(mode="in_place")


def test_retract_pause_requires_partial_rollout():
    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(["--rollout-batch-size", "1", "--rollout-weight-sync-pause-mode", "retract"])
    with pytest.raises(ValueError, match="requires --partial-rollout"):
        slime_validate_args(args)


def _unpatched_scheduler_source() -> str:
    return '''
class Scheduler:
    def is_fully_idle(self, for_health_check=False) -> bool:
        idle = True

        # Waiting queues: waiting + bootstrapping + preallocation + kv transfer (decode)
        idle &= len(self.waiting_queue) == 0
        return idle

    def pause_generation(self, recv_req):
        assert recv_req.mode in ("in_place", "retract")
        self._engine_paused = True
        retract_all([])

    def flush_cache(self, empty_cache: bool = True):
        """Flush memory pools (e.g., KV cache, Mamba cache) and optionally empty device allocator cache."""
        if self.is_fully_idle():
            self.cur_batch_for_debug = None
'''


def test_sglang_retract_flush_patch_is_idempotent_and_fail_closed(tmp_path: Path):
    scheduler = tmp_path / "scheduler.py"
    scheduler.write_text(_unpatched_scheduler_source(), encoding="utf-8")

    assert patch_file(scheduler) == "patched"
    patched = scheduler.read_text(encoding="utf-8")
    assert patched.count(_MARKER) == 2
    assert "ignore_waiting_queue=True" in patched
    assert patch_file(scheduler) == "patched"
    assert patch_file(scheduler, check_only=True) == "patched"

    unknown = tmp_path / "unknown.py"
    unknown.write_text("class Scheduler: pass\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match"):
        patch_file(unknown)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
