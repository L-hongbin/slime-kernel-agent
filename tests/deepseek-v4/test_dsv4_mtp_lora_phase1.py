"""Contract tests for LoRA + MTP spec-decode Phase 1 (adapter-on-target-only).

Upstream sglang hard-blocks LoRA with every neural-draft speculative algorithm
(server_args: "Currently LoRA is only compatible with NGRAM"). Phase 1 relaxes
that for our V4 MTP serving following the vLLM PR #11966 design: the adapter
applies to the target/verify forward only, and draft runners stay LoRA-free.
V4-Flash routes --speculative-algorithm NEXTN -> EAGLE (the NextN draft is not
a Gemma4 assistant, so _resolve_speculative_algorithm_alias returns "EAGLE"
and the EAGLE worker drives DeepseekV4ForCausalLMNextN as the draft).

The danger these tests pin: server_args is SHARED between the target and draft
ModelRunners. Any runtime LoRA gate that consults server_args.enable_lora
instead of the per-runner ModelRunner.lora_enabled either crashes the draft
(prepare_lora_batch on a runner with no lora_manager) or — worse — wraps the
draft/MTP modules with the target-shaped adapter: the same silent-corruption
class as the indexer wkv_gate leaf-name collision, with moving fault sites.

These are static source contracts against the patched fork (same pattern as
the launcher-wiring tests): they fail fast on a re-gating regression without
needing a GPU. The module skips wholesale on trees without the phase-1 patch
(production rollout nodes until the user approves rollout).
Patch: scripts/dsv4/patches/sglang_dsv4_mtp_lora_phase1.patch
"""

import os
import re
from pathlib import Path

import pytest

SGLANG_ROOT = Path(os.environ.get("V4_SGLANG_ROOT", "/sgl-workspace/sglang"))
SRT = SGLANG_ROOT / "python" / "sglang" / "srt"

if not (SRT / "server_args.py").exists():
    pytest.skip(f"sglang fork not present at {SGLANG_ROOT}", allow_module_level=True)

_MODEL_RUNNER = (SRT / "model_executor" / "model_runner.py").read_text()

if "self.lora_enabled" not in _MODEL_RUNNER:
    pytest.skip(
        "fork lacks the mtp-lora phase-1 patch (production baseline)",
        allow_module_level=True,
    )


def _read(rel: str) -> str:
    return (SRT / rel).read_text()


def test_spec_guard_allows_eagle_and_frozen_kv_mtp_only():
    text = _read("server_args.py")
    # Anchor to the LoRA-compat block via its error message so an unrelated
    # `not in [...]` list elsewhere can't produce a false match.
    m = re.search(
        r"if self\.speculative_algorithm not in \[(.*?)\]:" r"(?:(?!def ).)*?Currently LoRA is only compatible",
        text,
        flags=re.S,
    )
    assert m, "LoRA x speculative guard (with its error message) missing from server_args.py"
    allowed = {s.strip().strip('"') for s in m.group(1).split(",") if s.strip()}
    # NEXTN resolves before the guard runs (engine.check_server_args happens
    # after __post_init__'s handle_speculative_decoding), so post-resolution
    # names are the contract.
    assert allowed == {"NGRAM", "EAGLE", "FROZEN_KV_MTP", "None"}, (
        f"guard allow-list changed: {allowed}. EAGLE3/DFLASH/STANDALONE must "
        "stay blocked (unvalidated with LoRA); EAGLE (NEXTN post-alias) and "
        "FROZEN_KV_MTP are the validated adapter-on-target paths."
    )


def test_model_runner_defines_per_runner_lora_flag():
    assert re.search(
        r"self\.lora_enabled = server_args\.enable_lora and not is_draft_worker",
        _MODEL_RUNNER,
    ), (
        "ModelRunner must derive lora_enabled from enable_lora AND NOT "
        "is_draft_worker: draft runners share server_args with the target, so "
        "a shared-flag gate would init a LoRAManager on the draft and wrap "
        "the MTP block with the target-shaped adapter."
    )


def test_lora_manager_init_gated_per_runner():
    assert "if self.lora_enabled:\n            self.init_lora_manager()" in (
        _MODEL_RUNNER
    ), "init_lora_manager must be gated on the per-runner lora_enabled flag"


def test_draft_runner_wrap_audit_present():
    assert "elif server_args.enable_lora and self.is_draft_worker:" in _MODEL_RUNNER
    assert re.search(r"if wrapped:\s*\n(?:.*\n)*?\s*raise RuntimeError", _MODEL_RUNNER), (
        "draft runners must RAISE at launch if any module got LoRA-wrapped "
        "(RuntimeError, not assert — python -O strips asserts)"
    )


@pytest.mark.parametrize(
    "rel",
    [
        "model_executor/model_runner.py",
        "model_executor/forward_batch_info.py",
        "model_executor/cuda_graph_runner.py",
        "model_executor/piecewise_cuda_graph_runner.py",
        "model_executor/cpu_graph_runner.py",
    ],
)
def test_no_shared_flag_lora_gates_remain(rel):
    """Every runtime LoRA gate on the forward/capture path must consult the
    per-runner flag. The only allowed mentions of the shared flag in these
    files are the lora_enabled definition itself, the draft audit branch, and
    comments/strings."""
    text = _read(rel)
    offenders = []
    for i, line in enumerate(text.splitlines(), 1):
        code = line.split("#", 1)[0]
        # Catch the shared flag through any access path: attribute chains,
        # getattr, or the global getter (get_global_server_args().enable_lora).
        if re.search(r"(?<!lora_)enable_lora\b", code) and "lora_enabled" not in code:
            if "self.lora_enabled = server_args.enable_lora" in code:
                continue  # the per-runner flag definition
            if "elif server_args.enable_lora and self.is_draft_worker" in code:
                continue  # the draft wrap-audit branch
            offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, (
        "runtime LoRA gate(s) consult the SHARED server_args.enable_lora; on "
        "a draft runner these crash (no lora_manager) or corrupt (wrapped "
        f"draft modules): {offenders}"
    )


def test_draft_worker_flag_reaches_model_runner():
    text = _read("managers/tp_worker.py")
    assert re.search(r"is_draft_worker=self\.is_draft_worker", text), (
        "TpModelWorker must forward is_draft_worker into ModelRunner — all "
        "neural-draft spec workers (EAGLE v2, FrozenKVMTP) build their draft "
        "runner through TpModelWorker(is_draft_worker=True)."
    )
    # Chokepoint invariant: TpModelWorker is the ONLY ModelRunner construction
    # site in the fork; a second site could bypass the flag threading and the
    # per-runner gate with it.
    hits = []
    for py in SRT.rglob("*.py"):
        for i, line in enumerate(py.read_text().splitlines(), 1):
            if re.search(r"=\s*ModelRunner\(", line.split("#", 1)[0]):
                hits.append(f"{py.relative_to(SRT)}:{i}")
    assert hits == ["managers/tp_worker.py:357"] or (
        len(hits) == 1 and hits[0].startswith("managers/tp_worker.py")
    ), f"unexpected ModelRunner construction sites: {hits}"


def test_spec_workers_have_fail_closed_lora_audit():
    """Codex-review blocker remediation: the ModelRunner-side audit is gated
    on is_draft_worker, so it cannot catch that flag being lost. The spec
    workers know they built a draft regardless of the flag — they must raise
    (RuntimeError, not assert: -O strips asserts) if the draft runner came up
    LoRA-enabled."""
    for rel in ["speculative/eagle_worker_v2.py", "speculative/frozen_kv_mtp_worker.py"]:
        text = _read(rel)
        assert (
            "raise RuntimeError" in text and "target-only under speculative" in text
        ), f"{rel} missing the fail-closed draft LoRA audit"
        assert re.search(
            r"if server_args\.enable_lora and \(\s*getattr\(self\.draft_(model_)?runner",
            text,
        ), f"{rel}: audit must consult the draft runner directly, not the flag"
