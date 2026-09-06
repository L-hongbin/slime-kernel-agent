import logging
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.patch_sglang_qwen_mtp import REPLACEMENTS, patch_source

NUM_GPUS = 0


@pytest.mark.parametrize("path", list(REPLACEMENTS))
def test_patch_is_idempotent_and_rejects_unknown_source(path):
    old, new = REPLACEMENTS[path]
    assert patch_source(old, path) == new
    assert patch_source(new, path, check_only=True) == new
    with pytest.raises(RuntimeError):
        patch_source(old, path, check_only=True)
    with pytest.raises(RuntimeError):
        patch_source("unknown", path)


def _run_fragment(path, env):
    source = textwrap.dedent(REPLACEMENTS[path][1])
    exec("def run():\n" + textwrap.indent(source, "    "), env)
    env["run"]()


@pytest.mark.parametrize("cache_len", [None, 0])
def test_empty_mamba_cache_frees_only_owned_tail_and_never_inserts(cache_len):
    freed, released, unlocked = [], [], []
    req = SimpleNamespace(cache_protected_len=3, last_node=object())
    cache = SimpleNamespace(
        token_to_kv_pool_allocator=SimpleNamespace(free=freed.append),
        req_to_token_pool=SimpleNamespace(free_mamba_cache=released.append),
        dec_lock_ref=unlocked.append,
    )
    _run_fragment(
        "srt/mem_cache/mamba_radix_cache.py",
        {
            "self": cache,
            "cache_len": cache_len,
            "kv_indices": list(range(8)),
            "req": req,
        },
    )
    assert freed == [[3, 4, 5, 6, 7]]
    assert released == [req] and unlocked == [req.last_node]


def test_fa3_graph_table_has_draft_headroom():
    backend = SimpleNamespace(speculative_num_draft_tokens=4, page_size=64)
    _run_fragment(
        "srt/layers/attention/flashattention_backend.py",
        {
            "self": backend,
            "model_runner": SimpleNamespace(model_config=SimpleNamespace(context_len=40960)),
            "speculative_step_id": 0,
        },
    )
    assert backend.max_context_len == 40964
    assert backend.max_num_pages * backend.page_size >= 40964


@pytest.mark.parametrize("fail", [False, True])
def test_distributed_sync_dispatches_mtp_once_and_restores_loader(fail):
    target_calls, draft_calls = [], []
    target = SimpleNamespace(load_weights=target_calls.append)
    original_load = target.load_weights
    draft_cls = type("Qwen3_5ForCausalLMMTP", (), {"load_weights": lambda _, weights: draft_calls.append(weights)})
    runner = SimpleNamespace(model=draft_cls())
    weights = [("model.language_model.norm.weight", torch.ones(2)), ("mtp.fc.weight", torch.ones(2, 4))]

    def receive(_):
        target.load_weights(iter(weights))
        if fail:
            raise RuntimeError("receive failed")
        return True, "ok"

    manager = SimpleNamespace(
        tp_worker=SimpleNamespace(model_runner=SimpleNamespace(model=target), update_weights_from_distributed=receive),
        draft_worker=object(),
    )
    env = {
        "self": manager,
        "_get_draft_model_runner": lambda _: runner,
        "recv_req": None,
        "logger": logging.getLogger(__name__),
    }
    if fail:
        with pytest.raises(RuntimeError, match="receive failed"):
            _run_fragment("srt/managers/scheduler_components/weight_updater.py", env)
    else:
        _run_fragment("srt/managers/scheduler_components/weight_updater.py", env)
    assert target_calls == [[weights[0]]]
    assert draft_calls == [[weights[1]]]
    assert target.load_weights is original_load


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
