# Repository Guidelines

## Project Structure & Module Organization

`slime/` contains the core RL training framework, including rollout logic, Ray orchestration, backends, reward/filter hubs, and shared utilities. `slime_plugins/` contains optional integration modules for Megatron Bridge, model-specific support, and rollout buffer extensions. Entrypoints live at `train.py` and `train_async.py`; use the async entrypoint when training should overlap with rollout generation. `tests/` holds unit, integration, and CI-oriented tests, with reusable utility tests under `tests/utils/` and plugin contract tests under `tests/plugin_contracts/`. `examples/`, `scripts/`, and `docs/` provide runnable configurations, model launch scripts, and user/developer documentation. Static images are in `imgs/`.

## Build, Test, and Development Commands

- `pip install -e . --no-deps`: install slime from the repo checkout after preparing the project-specific CUDA/Megatron/SGLang environment.
- `bash build_conda.sh`: build the conda-based development environment when Docker is not suitable.
- `python train.py ...`: run the standard training loop using arguments from the selected example or script.
- `python train_async.py ...`: run the asynchronous training loop.
- `python -m pytest`: run the configured test suite under `tests/`.
- `python -m pytest tests/plugin_contracts`: run plugin API contract tests.
- `pre-commit run --all-files --show-diff-on-failure --color=always`: run formatting, import cleanup, and lint checks locally.

## Coding Style & Naming Conventions

Target Python 3.10+. Format Python with Black at 119 columns and sort imports with isort using the Black profile. Ruff enforces core `E`, `F`, `B`, and `UP` checks; do not bypass these without a clear reason. Prefer existing module patterns in `slime/` and `slime_plugins/` over new abstractions. Use `snake_case` for functions, modules, and test files; use `PascalCase` for classes.

## Testing Guidelines

Pytest is configured in `pyproject.toml` with strict markers and verbose output. Name Python tests `test_*.py` and mark broader scenarios with existing markers such as `unit`, `integration`, `system`, or `docs`. For plugin behavior, update or add tests in `tests/plugin_contracts/`. For training behavior, include a focused test or a reproducible command from `tests/` or `scripts/`.

## Commit & Pull Request Guidelines

Recent history uses short imperative subjects, often with scopes or prefixes such as `fix(qwen3_next): ...`, `[Fix] ...`, or `Add ...`. Keep commits focused and mention affected models, backends, or flags when useful. Pull requests should describe the bug or optimization, list verification commands, link related issues, and include benchmarks for performance changes. The project welcomes bug fixes and general large-scale RL optimizations with clear verification; avoid broad refactors or unverifiable features.
