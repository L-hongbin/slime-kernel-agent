def pytest_addoption(parser):
    group = parser.getgroup("cuda-kernel-eval")
    group.addoption(
        "--cuda-kernel-compiled",
        choices=("all", "true", "false"),
        default="all",
        help="Run only CUDA kernel eval cases with the selected compiled result.",
    )
