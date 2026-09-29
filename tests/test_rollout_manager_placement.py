import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

NUM_GPUS = 0
pytestmark = pytest.mark.unit


def _node(ip, *, alive=True, resource="slime_rollout", capacity=1):
    return {"Alive": alive, "NodeManagerAddress": ip, "Resources": {resource: capacity}}


@pytest.mark.parametrize("transport", ["object-store", "nixl"])
@pytest.mark.parametrize(
    "resource,nodes,expected_ip",
    [
        (None, [], None),
        (
            "slime_rollout",
            [
                _node("10.0.0.2"),
                _node("10.0.0.1", alive=False),
                _node("10.0.0.0", resource="slime_actor"),
                _node("10.0.0.3"),
                _node("10.0.0.4", capacity=0),
            ],
            "10.0.0.2",
        ),
        ("slime_rollout", [_node("10.0.0.3"), _node("10.0.0.2")], "10.0.0.2"),
    ],
)
def test_rollout_manager_placement(monkeypatch, transport, resource, nodes, expected_ip):
    from slime.ray import placement_group

    manager = object()
    actor = Mock()
    actor.options.return_value.remote.return_value = manager
    node_query = Mock(return_value=nodes)
    monkeypatch.setitem(sys.modules, "slime.ray.rollout", SimpleNamespace(RolloutManager=actor))
    monkeypatch.setattr(placement_group.ray, "nodes", node_query)
    monkeypatch.setattr(placement_group, "add_default_ray_env_vars", lambda: {"TEST_ENV": "1"})
    args = SimpleNamespace(
        rollout_data_transport=transport,
        rollout_placement_resource=resource,
        num_rollout=1,
        check_weight_update_equal=False,
        offload_rollout=False,
    )
    pg = object()
    assert placement_group.create_rollout_manager(args, pg) == (manager, None)
    actor.options.return_value.remote.assert_called_once_with(args, pg)
    options = actor.options.call_args.kwargs
    assert options["runtime_env"] == {"env_vars": {"TEST_ENV": "1"}}
    assert options["num_cpus"] == 1 and options["num_gpus"] == 0
    assert options.get("enable_tensor_transport", False) == (transport == "nixl")
    if resource:
        node_query.assert_called_once_with()
        assert options["resources"] == {f"node:{expected_ip}": 0.001}
    else:
        node_query.assert_not_called()
        assert "resources" not in options


@pytest.mark.parametrize(
    "nodes",
    [
        [],
        [_node("10.0.0.1", alive=False)],
        [_node("10.0.0.1", capacity=0)],
        [_node("10.0.0.1", resource="slime_actor")],
    ],
)
def test_rollout_manager_rejects_missing_live_resource(monkeypatch, nodes):
    from slime.ray import placement_group

    actor = Mock()
    monkeypatch.setitem(sys.modules, "slime.ray.rollout", SimpleNamespace(RolloutManager=actor))
    monkeypatch.setattr(placement_group.ray, "nodes", lambda: nodes)
    monkeypatch.setattr(placement_group, "add_default_ray_env_vars", lambda: {})
    args = SimpleNamespace(rollout_placement_resource="slime_rollout")
    with pytest.raises(RuntimeError, match="No Ray node provides rollout resource"):
        placement_group.create_rollout_manager(args, None)
    actor.options.assert_not_called()
