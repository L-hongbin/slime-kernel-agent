import os
import torch

ROUTING_REPLAY = None
ROUTING_REPLAY_LAYER_NOT_FOUND = object()


def set_routing_replay(replay):
    global ROUTING_REPLAY
    ROUTING_REPLAY = replay


class RoutingReplay:
    all_routing_replays = []

    def __init__(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []
        RoutingReplay.all_routing_replays.append(self)

    def record(self, top_indices):
        # offload top_indices to CPU pinned memory
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.top_indices_list.append(buf)

    def pop_forward(self):
        top_indices = self.top_indices_list[self.forward_index]
        self.forward_index += 1
        return top_indices.to(torch.cuda.current_device())

    def pop_backward(self):
        top_indices = self.top_indices_list[self.backward_index]
        self.backward_index += 1
        return top_indices.to(torch.cuda.current_device())

    def clear(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []

    def clear_forward(self):
        self.forward_index = 0

    @staticmethod
    def check_fully_consumed(context: str = "", *, model_modules=None) -> None:
        """Replay-consumption invariant (codex milestone review 2026-07-05).

        Every recorded routing entry must be popped exactly once by the train
        forward. Backward consumption follows the *actual* V4 decoder
        checkpoint plan: all layers for uniform recompute, the first K local
        layers for block recompute, and no layers when recompute is off. A
        same-shaped misalignment (wrong microbatch/layer replayed) advances the
        indices inconsistently, silently biasing gradients — the per-pop shape
        assert cannot catch it.

        ``model_modules`` is required so expectations are derived from concrete
        local layers and their Megatron recompute configuration.
        """
        if model_modules is None:
            raise ValueError("routing replay consumption checks require model_modules")
        backward_expectations = get_routing_replay_backward_expectations(model_modules)

        problems = []
        for i, replay in enumerate(RoutingReplay.all_routing_replays):
            n = len(replay.top_indices_list)
            expected_backward = n if backward_expectations[replay] else 0
            if replay.forward_index != n or replay.backward_index != expected_backward:
                problems.append(
                    f"replay[{i}]: recorded={n} forward_popped={replay.forward_index} "
                    f"backward_popped={replay.backward_index} "
                    f"backward_expected={expected_backward}"
                )
        if problems:
            msg = f"routing replay not fully consumed ({context}): " + "; ".join(problems[:4])
            raise AssertionError(msg)

    @staticmethod
    def clear_all():
        for replay in RoutingReplay.all_routing_replays:
            replay.clear()

    @staticmethod
    def clear_all_forward():
        for replay in RoutingReplay.all_routing_replays:
            replay.clear_forward()


def get_routing_replay_compute_topk(old_compute_topk):
    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            routing_replay_stage = os.environ["ROUTING_REPLAY_STAGE"]
            if routing_replay_stage == "fallthrough":
                return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
            if routing_replay_stage == "record":
                probs, top_indices = old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
                ROUTING_REPLAY.record(top_indices)
            elif routing_replay_stage == "replay_forward":
                top_indices = ROUTING_REPLAY.pop_forward()
                assert (
                    top_indices.shape[0] == scores.shape[0] and top_indices.shape[1] == topk
                ), f"[{torch.distributed.get_rank()}] top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                probs = scores.gather(1, top_indices)
            elif routing_replay_stage == "replay_backward":
                top_indices = ROUTING_REPLAY.pop_backward()
                assert (
                    top_indices.shape[0] == scores.shape[0] and top_indices.shape[1] == topk
                ), f"top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                probs = scores.gather(1, top_indices)
            return probs, top_indices
        else:
            return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

    return compute_topk


def register_routing_replay(module):
    if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
        module.routing_replay = RoutingReplay()

        def pre_forward_hook(*args, **kwargs):
            set_routing_replay(module.routing_replay)

        module.register_forward_pre_hook(pre_forward_hook)


def _iter_module_candidates(module):
    seen = set()
    stack = [module]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current

        for attr in ("module", "model", "language_model"):
            child = getattr(current, attr, None)
            if child is not None and child is not current:
                stack.append(child)

        modules = getattr(current, "modules", None)
        if callable(modules):
            for child in modules():
                if child is not current:
                    stack.append(child)


def get_routing_replay_backward_expectations(model_modules):
    """Return an explicit ``RoutingReplay -> bool`` backward-consumption plan.

    This mirrors ``_forward_v4_decoder_layers`` on the current PP/VP-local
    layer list. In block mode, only local indices ``[0, K)`` execute again in
    backward. Hash/dense layers have no replay object and therefore need no
    entry. Every replay recorded for the active pass must be mapped in block
    mode; empty registry entries from separately built ref/teacher models are
    ignored because they have nothing to consume.
    """
    registered = tuple(RoutingReplay.all_routing_replays)
    registered_set = set(registered)
    expectations = {replay: False for replay in registered}
    roots = model_modules if isinstance(model_modules, (list, tuple)) else (model_modules,)
    seen_candidates = set()
    mapped_replays = set()
    saw_block = False

    for root in roots:
        for candidate in _iter_module_candidates(root):
            if id(candidate) in seen_candidates:
                continue
            layer_ids = getattr(candidate, "layer_ids", None)
            layers = getattr(candidate, "layers", None)
            if layer_ids is None or layers is None:
                continue
            seen_candidates.add(id(candidate))

            config = getattr(candidate, "config", None)
            if getattr(config, "recompute_granularity", None) != "full":
                continue
            method = getattr(config, "recompute_method", None) or "uniform"
            if method not in ("uniform", "block"):
                raise ValueError("routing replay cannot resolve V4 recompute method; " f"got {method!r}")
            recompute_num_layers = getattr(config, "recompute_num_layers", None)
            if recompute_num_layers is None:
                recompute_num_layers = 1
            if isinstance(recompute_num_layers, bool) or not isinstance(recompute_num_layers, int):
                raise TypeError(
                    "routing replay requires integer V4 recompute_num_layers; " f"got {recompute_num_layers!r}"
                )
            if recompute_num_layers <= 0:
                raise ValueError(
                    "routing replay requires positive V4 recompute_num_layers; " f"got {recompute_num_layers}"
                )
            if method == "block" and recompute_num_layers > len(layers):
                raise ValueError(
                    "V4 block recompute_num_layers cannot exceed the local decoder "
                    f"layer count: K={recompute_num_layers}, local_layers={len(layers)}"
                )
            saw_block = saw_block or method == "block"

            for local_idx, layer in enumerate(layers):
                mlp = getattr(layer, "mlp", None)
                gate = getattr(mlp, "gate", None)
                replay = getattr(gate, "routing_replay", None)
                if replay not in registered_set:
                    continue
                mapped_replays.add(replay)
                expectations[replay] = method == "uniform" or local_idx < recompute_num_layers

    if saw_block:
        # The global registry may also contain routers from a separately built
        # ref/teacher/other model. Those objects legitimately remain empty in
        # this actor pass. Only a replay with recorded entries belongs to the
        # active pass and therefore must map to an explicit local-layer plan.
        recorded_replays = {replay for replay in registered if len(replay.top_indices_list) > 0}
        missing = recorded_replays - mapped_replays
        if missing:
            missing_indices = [i for i, replay in enumerate(registered) if replay in missing]
            raise AssertionError(
                "block recompute routing replay expectations could not map all "
                "recorded active replays to local decoder layers; "
                f"missing={missing_indices}"
            )
    return expectations


def get_rollout_routing_replay_for_layer(model_module, layer_id: int):
    """Return a layer gate replay object, None for a known non-replay layer, or sentinel."""
    for candidate in _iter_module_candidates(model_module):
        layer_ids = getattr(candidate, "layer_ids", None)
        layers = getattr(candidate, "layers", None)
        if layer_ids is None or layers is None:
            continue
        try:
            local_idx = tuple(layer_ids).index(layer_id)
        except ValueError:
            continue
        if local_idx >= len(layers):
            return None
        mlp = getattr(layers[local_idx], "mlp", None)
        gate = getattr(mlp, "gate", None)
        return getattr(gate, "routing_replay", None)
    return ROUTING_REPLAY_LAYER_NOT_FOUND


def record_rollout_routing_replay_for_layer(
    model_module,
    layer_id: int,
    layer_routed_experts: torch.Tensor,
    routing_replay_offset: int,
) -> int:
    replay = get_rollout_routing_replay_for_layer(model_module, layer_id)
    if replay is None:
        return routing_replay_offset
    if replay is ROUTING_REPLAY_LAYER_NOT_FOUND:
        if routing_replay_offset >= len(RoutingReplay.all_routing_replays):
            raise IndexError(
                "rollout routing replay offset out of range: "
                f"layer_id={layer_id}, offset={routing_replay_offset}, "
                f"registered_replays={len(RoutingReplay.all_routing_replays)}, "
                f"model_layer_ids={getattr(model_module, 'layer_ids', None)}"
            )
        replay = RoutingReplay.all_routing_replays[routing_replay_offset]
    replay.record(layer_routed_experts)
    return routing_replay_offset + 1


def should_skip_rollout_routing_replay_layer(model_module, layer_id: int) -> bool:
    """Return True for known layers whose router has no replay object."""
    replay = get_rollout_routing_replay_for_layer(model_module, layer_id)
    if replay is ROUTING_REPLAY_LAYER_NOT_FOUND:
        return False
    if replay is not None:
        return False
    layer_ids = getattr(model_module, "layer_ids", None)
    layers = getattr(model_module, "layers", None)
    if layer_ids is None or layers is None:
        return True
    try:
        local_idx = tuple(layer_ids).index(layer_id)
    except ValueError:
        return False
    if local_idx >= len(layers):
        return True
    mlp = getattr(layers[local_idx], "mlp", None)
    return bool(getattr(mlp, "is_hash", True))
