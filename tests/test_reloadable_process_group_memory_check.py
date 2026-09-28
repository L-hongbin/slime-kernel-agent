from __future__ import annotations

import pytest

from slime.utils import reloadable_process_group as rpg

NUM_GPUS = 0


@pytest.fixture(autouse=True)
def reset_check_timer(monkeypatch):
    monkeypatch.setattr(rpg, "_next_comm_memory_check", {})


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


def test_memory_checks_are_throttled_across_nested_wrappers(monkeypatch):
    times = iter([0.0, 0.01, 0.02, 0.11])
    calls = []
    monkeypatch.setattr(rpg.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(rpg, "available_memory", lambda: calls.append(1) or {"free_GB": 100})
    with rpg._wrap_low_level_call():
        with rpg._wrap_low_level_call():
            pass
    with rpg._wrap_low_level_call():
        pass
    with rpg._wrap_low_level_call():
        pass
    assert len(calls) == 2


def test_low_memory_is_rechecked_and_errors_always_reported(monkeypatch):
    calls = []
    monkeypatch.setattr(rpg.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(rpg, "available_memory", lambda: calls.append("check") or {"free_GB": 2})
    monkeypatch.setattr(rpg, "clear_memory", lambda: calls.append("clear"))
    monkeypatch.setattr(rpg, "print_memory", lambda _: calls.append("error") or {"free_GB": 2})
    for _ in range(2):
        with rpg._wrap_low_level_call():
            pass
    with pytest.raises(RuntimeError, match="collective failure") as failure:
        with rpg._wrap_low_level_call(check_memory=False):
            raise RuntimeError("collective failure")
    assert calls == ["check", "clear", "check", "clear", "error"]
    assert "mem_info=" in failure.value.__notes__[0]


def test_reload_resets_healthy_memory_check_timer(monkeypatch):
    pid = rpg.os.getpid()
    rpg._next_comm_memory_check[pid] = float("inf")
    monkeypatch.setattr(rpg.ReloadableProcessGroup, "GROUPS", {})
    rpg.ReloadableProcessGroup.reload_process_groups()
    assert pid not in rpg._next_comm_memory_check


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
