import os
import sys

from slime.backends.megatron_utils.path_bootstrap import ensure_megatron_lm_on_sys_path


def test_ensure_megatron_lm_on_sys_path_uses_env_candidate(monkeypatch, tmp_path):
    root = tmp_path / "Megatron-LM"
    (root / "megatron" / "training").mkdir(parents=True)
    monkeypatch.setenv("SLIME_MEGATRON_LM_PATH", str(root))
    monkeypatch.setenv("PYTHONPATH", "/existing")
    monkeypatch.setattr(sys, "path", ["/existing"])

    resolved = ensure_megatron_lm_on_sys_path(default_path=tmp_path / "missing")

    assert resolved == str(root)
    assert sys.path[0] == str(root)
    assert os.environ["PYTHONPATH"].split(os.pathsep)[:2] == [str(root), "/existing"]


def test_ensure_megatron_lm_on_sys_path_noops_without_training_tree(monkeypatch, tmp_path):
    monkeypatch.delenv("SLIME_MEGATRON_LM_PATH", raising=False)
    monkeypatch.setattr(sys, "path", ["/existing"])

    resolved = ensure_megatron_lm_on_sys_path(default_path=tmp_path / "missing")

    assert resolved is None
    assert sys.path == ["/existing"]
