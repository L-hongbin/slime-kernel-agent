import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Some modules under test (e.g. megatron_utils.model) import megatron.training,
# which lives in the Megatron-LM source tree, not the megatron-core install.
# No-op when Megatron-LM is absent.
from slime.backends.megatron_utils.path_bootstrap import ensure_megatron_lm_on_sys_path  # noqa: E402

ensure_megatron_lm_on_sys_path()


def pytest_addoption(parser):
    group = parser.getgroup("cuda-kernel-eval")
    group.addoption(
        "--cuda-kernel-compiled",
        choices=("all", "true", "false"),
        default="all",
        help="Run only CUDA kernel eval cases with the selected compiled result.",
    )
