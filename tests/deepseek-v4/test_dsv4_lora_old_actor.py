"""Offline tests for the V4 LoRA "old actor" adapter snapshot/swap (the cheap
``--keep-old-actor``). No GPU, no megatron model, no distributed.

Covers:
  1. enumerate_adapter_params picks exactly the trainable LoRA A/B tensors and
     skips the frozen base + non-adapter trainables.
  2. refresh() snapshots the live adapter and version-tags it.
  3. score_with_snapshot swaps the snapshot into the live model for the yielded
     forward, and restores the live adapter BIT-EXACT afterwards; the frozen base
     is never touched.
  4. Restore is exception-safe (try/finally): a crash inside the scoring block
     still restores the live adapter.
  5. The version-mismatch assert fires (score_with_snapshot expected_version and
     resolve_batch_gen_version).
  6. Optimizer state is undisturbed: swapping only mutates ``.data`` in place, the
     Parameter objects (that the optimizer keys on) keep their identity.

Run: python -m pytest tests/deepseek-v4/test_dsv4_lora_old_actor.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from slime.backends.megatron_utils.lora_old_actor import (
    LoRAOldActorSnapshot,
    enumerate_adapter_params,
    maybe_refresh_lora_old_actor_snapshot,
    resolve_batch_gen_version,
    should_recompute_old_actor_log_probs,
)

NUM_GPUS = 0


def _import_kernel_filter():
    """Import examples.kernel_agent.kernel_filter, defeating the Megatron-LM
    ``examples`` package that tests/conftest.py puts ahead of the repo on sys.path
    (its ``examples/`` has no ``kernel_agent``)."""
    import importlib

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))  # win over Megatron-LM's examples
    for mod in list(sys.modules):
        if mod == "examples" or mod.startswith("examples."):
            del sys.modules[mod]
    return importlib.import_module("examples.kernel_agent.kernel_filter")


class _FakeLoRALinear(nn.Module):
    """Mimics a megatron LinearAdapter: a frozen base weight + trainable A/B whose
    param names end with the ``.linear_in.weight`` / ``.linear_out.weight`` suffixes
    that ``is_adapter_param_name`` keys on."""

    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        base = nn.Linear(in_features, out_features, bias=False)
        base.weight.requires_grad_(False)  # frozen fp8/base stand-in
        self.base = base
        # Named so the fully-qualified param path ends in linear_in/out.weight.
        self.linear_in = nn.Linear(in_features, rank, bias=False)
        self.linear_out = nn.Linear(rank, out_features, bias=False)


def _build_model(seed: int = 0):
    """Two 'vp stages', each a module holding two adapter-wrapped linears plus a
    bare frozen param that must be ignored."""
    torch.manual_seed(seed)
    stages = []
    for _ in range(2):
        m = nn.Module()
        m.attn = _FakeLoRALinear(8, 8, rank=4)
        m.mlp = _FakeLoRALinear(8, 16, rank=4)
        # A non-adapter trainable param (e.g. a bias) must NOT be snapshotted.
        m.extra = nn.Parameter(torch.randn(8), requires_grad=True)
        stages.append(m)
    return stages


def _adapter_state(model) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in enumerate_adapter_params(model)}


def _perturb_adapter(model, scale: float = 0.1) -> None:
    with torch.no_grad():
        for _, p in enumerate_adapter_params(model):
            p.add_(scale * torch.randn_like(p))


def test_enumerate_selects_only_trainable_adapter_params():
    model = _build_model()
    names = [n for n, _ in enumerate_adapter_params(model)]
    # 2 stages x 2 modules x {linear_in, linear_out} = 8 adapter tensors.
    assert len(names) == 8
    assert all(n.endswith(".linear_in.weight") or n.endswith(".linear_out.weight") for n in names)
    # Frozen base + bare 'extra' bias are excluded.
    assert not any(".base." in n for n in names)
    assert not any(n.endswith(".extra") for n in names)


def test_freezing_base_hides_it_even_if_named_like_adapter():
    # requires_grad=False adapter-named params are excluded (base freeze wins).
    model = _build_model()
    with torch.no_grad():
        model[0].attn.linear_in.weight.requires_grad_(False)  # freeze one A tensor
    names = [n for n, _ in enumerate_adapter_params(model)]
    assert len(names) == 7  # 8 - 1 frozen out


def test_refresh_snapshots_and_tags_version():
    model = _build_model()
    snap = LoRAOldActorSnapshot(model, device="cpu", pin_memory=False)
    assert snap.has_params()
    assert snap.num_tensors == 8
    assert snap.version is None

    theta0 = _adapter_state(model)
    snap.refresh(version=1)
    assert snap.version == 1

    # Mutating the live adapter must not change the (copied) snapshot.
    _perturb_adapter(model)
    for name, buf in snap._snapshot.items():
        assert torch.equal(buf, theta0[name]), f"snapshot changed for {name}"


def test_fixed_debug_replay_seeds_once_and_skips_later_refreshes():
    model = _build_model()
    snap = LoRAOldActorSnapshot(model, device="cpu", pin_memory=False)
    theta0 = _adapter_state(model)

    assert maybe_refresh_lora_old_actor_snapshot(snap, version=0, freeze_after_seed=True)
    _perturb_adapter(model)
    theta1 = _adapter_state(model)

    assert not maybe_refresh_lora_old_actor_snapshot(snap, version=0, freeze_after_seed=True)
    with snap.score_with_snapshot(expected_version=0):
        scored = _adapter_state(model)
    assert all(torch.equal(scored[name], theta0[name]) for name in theta0)
    assert all(torch.equal(_adapter_state(model)[name], theta1[name]) for name in theta1)


def test_normal_debug_replay_refreshes_snapshot_after_seed():
    model = _build_model()
    snap = LoRAOldActorSnapshot(model, device="cpu", pin_memory=False)
    assert maybe_refresh_lora_old_actor_snapshot(snap, version=0, freeze_after_seed=False)
    _perturb_adapter(model)
    theta1 = _adapter_state(model)
    assert maybe_refresh_lora_old_actor_snapshot(snap, version=1, freeze_after_seed=False)
    assert snap.version == 1
    with snap.score_with_snapshot(expected_version=1):
        scored = _adapter_state(model)
    assert all(torch.equal(scored[name], theta1[name]) for name in theta1)


def test_score_swaps_in_snapshot_and_restores_bit_exact():
    model = _build_model()
    snap = LoRAOldActorSnapshot(model, device="cpu", pin_memory=False)

    theta_old = _adapter_state(model)  # behavioral policy θ_{k-1}
    snap.refresh(version=5)

    # Advance the live adapter to θ_k and remember it + the frozen base.
    _perturb_adapter(model)
    theta_cur = _adapter_state(model)
    # Key by (stage_idx, name): the same param path repeats across stages.
    base_before = {
        (i, n): p.detach().clone()
        for i, stage in enumerate(model)
        for n, p in stage.named_parameters()
        if ".base." in n
    }

    seen_inside = {}
    with snap.score_with_snapshot(expected_version=5):
        # Inside the scoring block the live model must hold the OLD adapter.
        seen_inside = _adapter_state(model)

    for name in theta_old:
        assert torch.equal(seen_inside[name], theta_old[name]), f"scoring did not load old adapter for {name}"
    # After the block, live adapter restored bit-exact to θ_k.
    theta_after = _adapter_state(model)
    for name in theta_cur:
        assert torch.equal(theta_after[name], theta_cur[name]), f"restore not bit-exact for {name}"
    # Frozen base never touched.
    for i, stage in enumerate(model):
        for n, p in stage.named_parameters():
            if ".base." in n:
                assert torch.equal(p.detach(), base_before[(i, n)])


def test_restore_is_exception_safe():
    model = _build_model()
    snap = LoRAOldActorSnapshot(model, device="cpu", pin_memory=False)
    snap.refresh(version=2)
    _perturb_adapter(model)
    theta_cur = _adapter_state(model)

    class _Boom(RuntimeError):
        pass

    with pytest.raises(_Boom):
        with snap.score_with_snapshot(expected_version=2):
            raise _Boom("scoring forward crashed")

    # Live adapter must still be θ_k despite the crash mid-swap.
    theta_after = _adapter_state(model)
    for name in theta_cur:
        assert torch.equal(theta_after[name], theta_cur[name]), f"live adapter not restored after crash: {name}"


def test_version_mismatch_assert_fires():
    model = _build_model()
    snap = LoRAOldActorSnapshot(model, device="cpu", pin_memory=False)
    snap.refresh(version=7)
    with pytest.raises(RuntimeError, match="version mismatch"):
        with snap.score_with_snapshot(expected_version=6):
            pass


def test_score_requires_seeded_snapshot():
    model = _build_model()
    snap = LoRAOldActorSnapshot(model, device="cpu", pin_memory=False)
    with pytest.raises(AssertionError):
        with snap.score_with_snapshot(expected_version=1):
            pass


def test_optimizer_state_undisturbed_by_swap():
    # Param object identity must survive the swap so optimizer state (keyed on the
    # Parameter) is not orphaned. Only .data is mutated in place.
    model = _build_model()
    params = [p for _, p in enumerate_adapter_params(model)]
    ids_before = {id(p) for p in params}
    versions_before = {id(p): p._version for p in params}

    snap = LoRAOldActorSnapshot(model, device="cpu", pin_memory=False)
    snap.refresh(version=3)
    _perturb_adapter(model)
    with snap.score_with_snapshot(expected_version=3):
        pass

    params_after = [p for _, p in enumerate_adapter_params(model)]
    assert {id(p) for p in params_after} == ids_before
    # in-place .data copy bumps the tensor _version but keeps the object; assert we
    # at least did not replace the Parameter (identity preserved above).
    for p in params_after:
        assert p._version >= versions_before[id(p)]


def test_resolve_batch_gen_version_uniform_and_mismatch():
    # Uniform batch -> returns the single version (no dist initialized -> local path).
    assert resolve_batch_gen_version([5, 5, 5], snapshot_version=5) == 5
    # FAIL-CLOSED: missing/unstamped/mixed provenance is a hard error under
    # keep-old-actor (an unstamped sample's behavioral version is unknown).
    with pytest.raises(RuntimeError, match="fail-closed"):
        resolve_batch_gen_version([None, None], snapshot_version=5)
    with pytest.raises(RuntimeError, match="fail-closed"):
        resolve_batch_gen_version([], snapshot_version=5)
    with pytest.raises(RuntimeError, match="fail-closed"):
        resolve_batch_gen_version(None, snapshot_version=5)
    with pytest.raises(RuntimeError, match="fail-closed"):
        resolve_batch_gen_version([5, None, 5], snapshot_version=5)
    # Any mismatch -> raise loudly.
    with pytest.raises(RuntimeError, match="version check failed"):
        resolve_batch_gen_version([5, 6], snapshot_version=5)
    with pytest.raises(RuntimeError, match="version check failed"):
        resolve_batch_gen_version([4, 4], snapshot_version=5)


def test_no_params_model_reports_empty():
    m = nn.Module()
    m.w = nn.Parameter(torch.randn(4), requires_grad=True)  # not adapter-named
    snap = LoRAOldActorSnapshot([m], device="cpu", pin_memory=False)
    assert not snap.has_params()
    assert snap.num_tensors == 0


# --- TIS plumbing (req 4a) ------------------------------------------------------


def test_vanilla_tis_ratio_is_old_over_rollout_with_clipping():
    """With --keep-old-actor, TIS train_log_probs == old-actor recompute (behavioral
    θ_V) and rollout_log_probs == sglang θ_V, so the weight is exp(old − rollout) =
    the pure train/infer (cross-engine) mismatch, clipped to [low, high]."""
    from argparse import Namespace

    from slime.backends.megatron_utils.loss import vanilla_tis_function

    args = Namespace(tis_clip_low=0.5, tis_clip=2.0)
    old = torch.tensor([0.0, 1.0, -3.0, 0.1])  # behavioral old-actor log-probs
    rollout = torch.tensor([0.0, 0.0, 0.0, 0.0])  # sglang log-probs
    pg_loss = torch.ones(4)
    masks = [torch.ones(4)]

    out_loss, out_masks, metrics = vanilla_tis_function(
        args,
        pg_loss=pg_loss.clone(),
        train_log_probs=[old],
        rollout_log_probs=[rollout],
        loss_masks=masks,
    )
    expected_tis = torch.exp(old - rollout)
    assert torch.allclose(metrics["tis"], expected_tis)
    # index 2: exp(-3) ~ 0.0498 clamps up to low=0.5; index 1: exp(1)=2.718 clamps to 2.0.
    expected_w = torch.clamp(expected_tis, min=0.5, max=2.0)
    assert torch.allclose(out_loss, pg_loss * expected_w)
    assert out_masks is masks


# --- same-stack MIS option (req 4b) --------------------------------------------


def test_sequence_mis_ratio_source_selects_pair(monkeypatch):
    """ratio_source='old_actor' feeds (cur_log_probs, log_probs) — a same-stack
    megatron drift ratio — to the aggregator; default feeds (log_probs,
    rollout_log_probs), the cross-engine pair."""
    from argparse import Namespace

    kf = _import_kernel_filter()

    captured = {}

    def fake_batch(*, train_log_probs, rollout_log_probs, qkv_format, max_seq_lens, **kwargs):
        captured["train"] = train_log_probs
        captured["rollout"] = rollout_log_probs
        captured["qkv_format"] = qkv_format
        captured["max_seq_lens"] = max_seq_lens

    monkeypatch.setattr(kf, "_batch_sequence_mis", fake_batch)

    old = [torch.tensor([0.0, 0.0])]
    cur = [torch.tensor([0.1, 0.2])]
    sglang = [torch.tensor([-0.5, -0.5])]

    def make_args(ratio_source):
        return Namespace(
            sequence_mis_aggregation="geometric",
            sequence_mis_lower=0.5,
            sequence_mis_upper=2.0,
            sequence_mis_token_veto_threshold=None,
            sequence_mis_use_advantage=False,
            sequence_mis_mode="batch",
            sequence_mis_ratio_source=ratio_source,
            max_turns=None,
            qkv_format="bshd",
        )

    def make_data(with_cur):
        d = {
            "log_probs": [t.clone() for t in old],
            "rollout_log_probs": [t.clone() for t in sglang],
            "loss_masks": [torch.ones(2)],
            "total_lengths": [2],
            "response_lengths": [2],
            "max_seq_lens": [256],
        }
        if with_cur:
            d["cur_log_probs"] = [t.clone() for t in cur]
        return d

    # default: cross-engine pair
    kf.sequence_mis(make_args("rollout"), 0, make_data(with_cur=False))
    assert torch.equal(captured["train"][0], old[0])
    assert torch.equal(captured["rollout"][0], sglang[0])
    assert captured["qkv_format"] == "bshd"
    assert captured["max_seq_lens"] == [256]

    # old_actor: same-stack pair (cur numerator, old denominator)
    kf.sequence_mis(make_args("old_actor"), 0, make_data(with_cur=True))
    assert torch.equal(captured["train"][0], cur[0])
    assert torch.equal(captured["rollout"][0], old[0])


@pytest.mark.parametrize("mode", ["loop", "batch"])
def test_sequence_mis_threads_bshd_layout_to_cp_gather(monkeypatch, mode):
    """Both implementations must gather with the physical BSHD width."""
    from argparse import Namespace

    kf = _import_kernel_filter()
    calls = []

    def fake_gather(tensor, total_length, response_length, *, qkv_format, max_seq_len):
        calls.append((total_length, response_length, qkv_format, max_seq_len))
        return tensor

    monkeypatch.setattr(kf, "all_gather_with_cp", fake_gather)
    args = Namespace(
        sequence_mis_aggregation="geometric",
        sequence_mis_lower=0.5,
        sequence_mis_upper=2.0,
        sequence_mis_token_veto_threshold=None,
        sequence_mis_use_advantage=False,
        sequence_mis_mode=mode,
        sequence_mis_batch_size=8,
        sequence_mis_ratio_source="rollout",
        n_samples_per_prompt=1,
        max_turns=None,
        qkv_format="bshd",
    )
    data = {
        "log_probs": [torch.tensor([0.0, 0.0])],
        "rollout_log_probs": [torch.tensor([0.0, 0.0])],
        "loss_masks": [torch.ones(2)],
        "total_lengths": [3],
        "response_lengths": [2],
        "max_seq_lens": [256],
    }

    kf.sequence_mis(args, 0, data)

    assert calls == [(3, 2, "bshd", 256), (3, 2, "bshd", 256)]


def test_sequence_mis_old_actor_requires_cur_log_probs():
    from argparse import Namespace

    kf = _import_kernel_filter()

    args = Namespace(
        sequence_mis_aggregation="geometric",
        sequence_mis_lower=0.5,
        sequence_mis_upper=2.0,
        sequence_mis_token_veto_threshold=None,
        sequence_mis_use_advantage=False,
        sequence_mis_mode="batch",
        sequence_mis_ratio_source="old_actor",
        max_turns=None,
    )
    data = {
        "log_probs": [torch.zeros(2)],
        "rollout_log_probs": [torch.zeros(2)],
        "loss_masks": [torch.ones(2)],
        "total_lengths": [2],
        "response_lengths": [2],
    }  # cur_log_probs missing
    with pytest.raises(ValueError, match="cur_log_probs"):
        kf.sequence_mis(args, 0, data)


def test_parse_sequence_mis_ratio_source():
    from argparse import Namespace

    from slime.utils.arguments import _parse_sequence_mis_args

    args = Namespace(
        sequence_mis_config='{"aggregation":"geometric","lower":0.5,"upper":2.0,"ratio_source":"old_actor"}',
        max_turns=1,
    )
    _parse_sequence_mis_args(args)
    assert args.sequence_mis_ratio_source == "old_actor"

    bad = Namespace(sequence_mis_config='{"ratio_source":"bogus"}', max_turns=1)
    with pytest.raises(ValueError, match="ratio_source"):
        _parse_sequence_mis_args(bad)


# --- V3: async version-bookkeeping simulation ---------------------------------


def test_v3_async_version_bookkeeping_simulation():
    """Transcribe train_async.py's generate/update schedule (interval=1) to drive
    the snapshot exactly as MegatronTrainRayActor.train_actor does, and assert at
    every scored batch: (a) resolve_batch_gen_version accepts it (snapshot version
    == batch gen_weight_version), and (b) the adapter swapped in equals the true
    BEHAVIORAL policy (the θ the engine served for that version)."""
    model = _build_model()
    snap = LoRAOldActorSnapshot(model, device="cpu", pin_memory=False)

    def set_theta(v: int) -> None:  # write a scalar into every adapter param
        with torch.no_grad():
            for _, p in enumerate_adapter_params(model):
                p.fill_(float(v))

    def live_theta() -> float:
        return float(enumerate_adapter_params(model)[0][1].flatten()[0])

    class _WU:
        weight_version = 0

    wu = _WU()
    pushed: dict[int, float] = {}  # engine version -> θ it serves (== behavioral policy)

    def update_weights() -> None:
        # weight_updater.update_weights(): bump version, then push the LIVE adapter.
        wu.weight_version += 1
        pushed[wu.weight_version] = live_theta()

    theta = 0
    set_theta(theta)  # θ_0
    update_weights()  # pre-loop push (train_async.py:29): wv=1, pushed[1]=θ_0

    def generate() -> dict:  # stamps the batch with the CURRENT engine version
        return {"gen_weight_versions": [wu.weight_version, wu.weight_version]}

    NUM = 6
    observed_expected_v = []
    next_batch = generate()  # gen(start_rollout_id, v=weight_version)
    for rollout_id in range(NUM):
        if next_batch is not None:
            curr = next_batch
        if rollout_id + 1 < NUM:
            next_batch = generate()  # "start next rollout early" (v=weight_version)

        # ---- train_actor(curr) ----
        if snap.version is None:  # seed on the first step
            snap.refresh(version=wu.weight_version)
        expected_v = resolve_batch_gen_version(curr["gen_weight_versions"], snap.version)
        observed_expected_v.append(expected_v)
        behavioral = pushed[expected_v]
        with snap.score_with_snapshot(expected_version=expected_v):
            assert (
                live_theta() == behavioral
            ), f"iter {rollout_id}: scored under θ={live_theta()} but behavioral policy is θ={behavioral}"
        snap.refresh(version=wu.weight_version)  # refresh to θ_k BEFORE the gradient step
        theta += 1
        set_theta(theta)  # gradient step -> θ_{k+1}

        # ---- end of iteration (interval == 1): consume the early gen, then push ----
        leftover = next_batch
        next_batch = None
        update_weights()
        next_batch = leftover  # becomes next iteration's curr

    # Derived schedule: iters 0,1 train v1 (θ_0); iter k>=2 trains v_k (θ_{k-1}).
    assert observed_expected_v == [1, 1, 2, 3, 4, 5], observed_expected_v


def test_v3_buffered_stale_batch_trips_assert():
    """A buffered carry-over (gen_weight_version older than the snapshot) must fail
    loudly rather than be scored with the wrong behavior weights."""
    model = _build_model()
    snap = LoRAOldActorSnapshot(model, device="cpu", pin_memory=False)
    snap.refresh(version=4)  # current behavioral snapshot is v4
    # A batch mixing a fresh v4 sample with a stale v2 carry-over.
    with pytest.raises(RuntimeError, match="version check failed"):
        resolve_batch_gen_version([4, 4, 2], snap.version)


def test_v3_tis_finite_and_centered_near_one():
    """With --keep-old-actor the TIS weight is exp(old_actor_V − sglang_V) — the pure
    train/infer mismatch at one policy version. On realistic ~0.04-nat/token mismatch
    it is finite and centered near 1 (well inside the clip band)."""
    from argparse import Namespace

    from slime.backends.megatron_utils.loss import vanilla_tis_function

    torch.manual_seed(0)
    n = 4096
    rollout = torch.randn(n) * 0.5  # sglang log-probs
    old = rollout + torch.randn(n) * 0.04  # megatron old-actor recompute (small mismatch)
    args = Namespace(tis_clip_low=0.5, tis_clip=2.0)
    _, _, metrics = vanilla_tis_function(
        args, pg_loss=torch.ones(n), train_log_probs=[old], rollout_log_probs=[rollout], loss_masks=[torch.ones(n)]
    )
    tis = metrics["tis"]
    assert torch.isfinite(tis).all()
    assert 0.98 < tis.mean().item() < 1.02, tis.mean().item()
    # Essentially nothing clips at this mismatch level.
    assert metrics["tis_clipfrac"].mean().item() < 0.01


def test_validate_sequence_mis_ratio_source_cross_checks():
    from argparse import Namespace

    from slime.utils.arguments import _validate_sequence_mis_ratio_source

    # default ratio_source -> no constraints
    _validate_sequence_mis_ratio_source(Namespace(sequence_mis_ratio_source="rollout"))

    # old_actor requires keep_old_actor
    with pytest.raises(ValueError, match="keep-old-actor"):
        _validate_sequence_mis_ratio_source(
            Namespace(sequence_mis_ratio_source="old_actor", keep_old_actor=False, use_routing_replay=False)
        )

    # old_actor incompatible with routing replay
    with pytest.raises(ValueError, match="routing replay"):
        _validate_sequence_mis_ratio_source(
            Namespace(
                sequence_mis_ratio_source="old_actor",
                keep_old_actor=True,
                use_routing_replay=True,
                use_rollout_routing_replay=True,
            )
        )
    with pytest.raises(ValueError, match="routing replay"):
        _validate_sequence_mis_ratio_source(
            Namespace(
                sequence_mis_ratio_source="old_actor",
                keep_old_actor=True,
                use_routing_replay=False,
                use_rollout_routing_replay=True,
            )
        )

    # valid combination
    _validate_sequence_mis_ratio_source(
        Namespace(
            sequence_mis_ratio_source="old_actor",
            keep_old_actor=True,
            use_routing_replay=False,
            use_rollout_routing_replay=False,
        )
    )


def test_debug_freeze_old_actor_snapshot_validation_is_fail_closed():
    from argparse import Namespace

    from slime.utils.arguments import _validate_debug_freeze_old_actor_snapshot

    # Off is inert in every mode.
    _validate_debug_freeze_old_actor_snapshot(Namespace(debug_freeze_old_actor_snapshot=False))

    valid = dict(
        debug_freeze_old_actor_snapshot=True,
        debug_train_only=True,
        load_debug_rollout_data="/tmp/frozen.pt",
        keep_old_actor=True,
    )
    _validate_debug_freeze_old_actor_snapshot(Namespace(**valid))

    for missing_field, expected_flag in (
        ("debug_train_only", "--debug-train-only"),
        ("load_debug_rollout_data", "--load-debug-rollout-data"),
        ("keep_old_actor", "--keep-old-actor"),
    ):
        kwargs = dict(valid)
        kwargs[missing_field] = None if missing_field == "load_debug_rollout_data" else False
        with pytest.raises(ValueError, match=expected_flag):
            _validate_debug_freeze_old_actor_snapshot(Namespace(**kwargs))


def test_denominator_ab_can_force_the_same_old_actor_forward_path():
    from argparse import Namespace

    assert should_recompute_old_actor_log_probs(
        Namespace(
            use_rollout_logprobs=True,
            get_mismatch_metrics=False,
            debug_force_old_actor_logprob_recompute=True,
        )
    )
    assert should_recompute_old_actor_log_probs(
        Namespace(
            use_rollout_logprobs=False,
            get_mismatch_metrics=False,
            debug_force_old_actor_logprob_recompute=True,
        )
    )
    assert not should_recompute_old_actor_log_probs(
        Namespace(
            use_rollout_logprobs=True,
            get_mismatch_metrics=False,
            debug_force_old_actor_logprob_recompute=False,
        )
    )


def test_debug_force_old_actor_logprob_recompute_validation_is_fail_closed():
    from argparse import Namespace

    from slime.utils.arguments import _validate_debug_force_old_actor_logprob_recompute

    valid = dict(
        debug_force_old_actor_logprob_recompute=True,
        debug_train_only=True,
        load_debug_rollout_data="/tmp/frozen.pt",
        keep_old_actor=True,
    )
    _validate_debug_force_old_actor_logprob_recompute(Namespace(**valid))

    for missing_field, expected_flag in (
        ("debug_train_only", "--debug-train-only"),
        ("load_debug_rollout_data", "--load-debug-rollout-data"),
        ("keep_old_actor", "--keep-old-actor"),
    ):
        kwargs = dict(valid)
        kwargs[missing_field] = None if missing_field == "load_debug_rollout_data" else False
        with pytest.raises(ValueError, match=expected_flag):
            _validate_debug_force_old_actor_logprob_recompute(Namespace(**kwargs))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
