from argparse import Namespace
from pathlib import Path
import sys

import pytest

MEGATRON_PATH = Path("/root/Megatron-LM")
if MEGATRON_PATH.exists():
    sys.path.insert(0, str(MEGATRON_PATH))

model_utils = pytest.importorskip("slime.backends.megatron_utils.model")


class FakeHandle:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class FakeModule:
    def __init__(self, children=None):
        self.children = children or []

    def modules(self):
        yield self
        for child in self.children:
            yield from child.modules()

    def register_forward_pre_hook(self, _hook):
        return FakeHandle()


class FakeDDP:
    def __init__(self):
        self.module = FakeModule(children=[FakeModule()])
        self.remove_forward_pre_hook_handles = {}
        self.disable_calls = 0
        self.enable_calls = 0
        self.param_sync_calls = 0
        self.enable_forward_pre_hook()

    def enable_forward_pre_hook(self):
        assert len(self.remove_forward_pre_hook_handles) == 0
        self.enable_calls += 1
        for module in self.module.modules():
            self.remove_forward_pre_hook_handles[module] = module.register_forward_pre_hook(None)

    def disable_forward_pre_hook(self, param_sync=True):
        self.disable_calls += 1
        for module in self.module.modules():
            handle = self.remove_forward_pre_hook_handles[module]
            handle.remove()
            del self.remove_forward_pre_hook_handles[module]
        if param_sync:
            self.start_param_sync(force_sync=True)

    def start_param_sync(self, force_sync=False):
        assert force_sync
        self.param_sync_calls += 1


@pytest.fixture(autouse=True)
def fake_ddp(monkeypatch):
    monkeypatch.setattr(model_utils, "DDP", FakeDDP)


@pytest.mark.unit
def test_disable_forward_pre_hook_is_idempotent():
    chunk = FakeDDP()

    assert model_utils.disable_forward_pre_hook([chunk], param_sync=False)
    assert chunk.remove_forward_pre_hook_handles == {}
    assert chunk.disable_calls == 1

    assert not model_utils.disable_forward_pre_hook([chunk], param_sync=False)
    assert chunk.disable_calls == 1


@pytest.mark.unit
def test_enable_forward_pre_hook_repairs_partial_handle_state():
    chunk = FakeDDP()
    first_handle = next(iter(chunk.remove_forward_pre_hook_handles.values()))
    first_module = next(iter(chunk.remove_forward_pre_hook_handles))
    chunk.remove_forward_pre_hook_handles = {first_module: first_handle}

    model_utils.enable_forward_pre_hook([chunk])

    assert first_handle.removed
    assert len(chunk.remove_forward_pre_hook_handles) == 2
    assert chunk.enable_calls == 2


@pytest.mark.unit
def test_save_preserves_already_disabled_forward_pre_hook(monkeypatch):
    chunk = FakeDDP()
    model_utils.disable_forward_pre_hook([chunk], param_sync=False)
    save_calls = []

    monkeypatch.setattr(
        model_utils,
        "get_args",
        lambda: Namespace(use_distributed_optimizer=True, overlap_param_gather=True),
    )
    monkeypatch.setattr(model_utils, "save_checkpoint", lambda *args, **kwargs: save_calls.append((args, kwargs)))

    model_utils.save(1, [chunk], optimizer=None, opt_param_scheduler=None)

    assert len(save_calls) == 1
    assert chunk.remove_forward_pre_hook_handles == {}
