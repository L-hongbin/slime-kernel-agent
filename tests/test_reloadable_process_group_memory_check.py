from __future__ import annotations

import pytest

from slime.utils import reloadable_process_group as rpg

NUM_GPUS = 0


@pytest.fixture(autouse=True)
def reset_memory_poll_state(monkeypatch):
    monkeypatch.setattr(rpg, "_COMM_MEMORY_CHECK_INTERVAL", 1)
    monkeypatch.setattr(rpg, "_comm_memory_checks_remaining", 0)
    monkeypatch.setattr(rpg, "_comm_memory_check_deadline", 0.0)


@pytest.mark.unit
def test_memory_polling_is_bounded_by_calls_and_time(monkeypatch):
    calls = []
    clock = [1.0]
    monkeypatch.setattr(rpg, "_COMM_MEMORY_CHECK_INTERVAL", 3)
    monkeypatch.setattr(rpg.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(rpg, "available_memory", lambda: calls.append(clock[0]) or {"free_GB": 20})
    for _ in range(4):
        with rpg._wrap_low_level_call():
            pass
    assert calls == [1.0, 1.0]
    clock[0] = 1.3
    with rpg._wrap_low_level_call():
        pass
    assert calls == [1.0, 1.0, 1.3]


@pytest.mark.unit
def test_low_memory_keeps_polling_every_call(monkeypatch):
    calls = []
    monkeypatch.setattr(rpg, "_COMM_MEMORY_CHECK_INTERVAL", 64)
    monkeypatch.setattr(rpg, "available_memory", lambda: calls.append("poll") or {"free_GB": 2})
    monkeypatch.setattr(rpg, "clear_memory", lambda: calls.append("clear"))
    for _ in range(3):
        with rpg._wrap_low_level_call():
            pass
    assert calls == ["poll", "clear"] * 3


@pytest.mark.unit
def test_throttled_memory_polling_keeps_exception_diagnostics(monkeypatch):
    monkeypatch.setattr(rpg, "_COMM_MEMORY_CHECK_INTERVAL", 64)
    monkeypatch.setattr(rpg, "available_memory", lambda: {"free_GB": 20})
    monkeypatch.setattr(rpg, "print_memory", lambda message: {"free_GB": 19})
    with pytest.raises(RuntimeError, match="collective failed") as error:
        with rpg._wrap_low_level_call():
            raise RuntimeError("collective failed")
    assert "free_GB" in error.value.__notes__[0]


@pytest.mark.unit
def test_selected_comm_ops_skip_memory_check():
    skipped_ops = {
        "all_gather_into_tensor",
        "allgather_into_tensor_coalesced",
        "barrier",
        "broadcast_object_list",
        "reduce_scatter_tensor",
        "all_to_all_single",
        "isend",
        "irecv",
    }
    checked_ops = {
        "all_reduce",
        "all_gather",
        "broadcast",
        "reduce_scatter",
        "all_to_all",
        "send",
        "recv",
        "reduce_scatter_tensor_coalesced",
    }

    for op_name in skipped_ops:
        assert not rpg._should_check_memory_for_comm(op_name)

    for op_name in checked_ops:
        assert rpg._should_check_memory_for_comm(op_name)


@pytest.mark.unit
def test_wrap_low_level_call_can_skip_available_memory(monkeypatch):
    calls = []

    def fake_available_memory():
        calls.append("available_memory")
        return {"free_GB": 100}

    monkeypatch.setattr(rpg, "available_memory", fake_available_memory)

    with rpg._wrap_low_level_call(check_memory=False):
        pass

    assert calls == []


@pytest.mark.unit
def test_wrap_low_level_call_checks_available_memory_by_default(monkeypatch):
    calls = []

    def fake_available_memory():
        calls.append("available_memory")
        return {"free_GB": 100}

    monkeypatch.setattr(rpg, "available_memory", fake_available_memory)

    with rpg._wrap_low_level_call():
        pass

    assert calls == ["available_memory"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
