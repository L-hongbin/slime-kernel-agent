import json
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from slime.backends.megatron_utils.update_weight import update_weight_from_distributed as distributed_update
from slime.backends.sglang_utils.sglang_engine import SGLangEngine

NUM_GPUS = 0


@dataclass
class _DeltaParam:
    name: str
    value: int


def test_sglang_engine_serializes_delta_spec_for_http_request():
    engine = SGLangEngine.__new__(SGLangEngine)
    captured = {}
    engine._make_request = lambda endpoint, payload: captured.update(endpoint=endpoint, payload=payload)
    delta = SimpleNamespace(
        encoding=SimpleNamespace(value="overwrite"),
        params=[_DeltaParam(name="weight", value=3)],
        checksum=7,
    )

    engine.update_weights_from_distributed(
        names=["weight"],
        dtypes=[torch.bfloat16],
        shapes=[torch.Size([2, 3])],
        group_name="group",
        delta=delta,
        load_format="delta",
    )

    assert captured["endpoint"] == "update_weights_from_distributed"
    assert captured["payload"]["dtypes"] == ["bfloat16"]
    assert json.loads(captured["payload"]["delta"]) == {
        "encoding": "overwrite",
        "params": [{"name": "weight", "value": 3}],
        "checksum": 7,
    }


def test_full_sync_does_not_send_optional_delta_keywords(monkeypatch):
    calls = []

    class _RemoteMethod:
        @staticmethod
        def remote(**kwargs):
            calls.append(kwargs)
            return "ref"

    class _Engine:
        update_weights_from_distributed = _RemoteMethod()

    class _Handle:
        def wait(self):
            return None

    monkeypatch.setattr(distributed_update.dist, "broadcast", lambda *args, **kwargs: _Handle())
    tensor = torch.zeros(2)

    refs = distributed_update.update_weights_from_distributed(
        group_name="group",
        group=object(),
        weight_version=4,
        rollout_engines=[_Engine()],
        converted_named_tensors=[("weight", tensor)],
    )

    assert refs == ["ref"]
    assert calls[0]["weight_version"] == "4"
    assert "load_format" not in calls[0]
    assert "delta" not in calls[0]
