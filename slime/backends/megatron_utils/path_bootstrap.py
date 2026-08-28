import os
import sys
from pathlib import Path


def ensure_megatron_lm_on_sys_path(default_path: str | os.PathLike[str] = "/root/Megatron-LM") -> str | None:
    """Make Megatron-LM's full source tree importable when only megatron-core is installed."""
    candidates: list[str | os.PathLike[str]] = []
    env_path = os.environ.get("SLIME_MEGATRON_LM_PATH")
    if env_path:
        candidates.extend(path for path in env_path.split(os.pathsep) if path)
    if default_path:
        candidates.append(default_path)

    for candidate in candidates:
        root = Path(candidate).expanduser()
        if not (root / "megatron" / "training").is_dir():
            continue

        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        pythonpath = os.environ.get("PYTHONPATH")
        pythonpath_entries = pythonpath.split(os.pathsep) if pythonpath else []
        if root_str not in pythonpath_entries:
            os.environ["PYTHONPATH"] = os.pathsep.join([root_str, *pythonpath_entries])
        return root_str

    return None
