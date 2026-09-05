---
name: add-tests-and-ci
description: Guide for adding or updating slime tests and CI wiring. Use when tasks require new test cases, CI registration, test matrix updates, or workflow template changes.
---

# Add Tests and CI

## Test execution contract

- CI executes registered test files with `python tests/<file>.py`, not only pytest discovery. New CPU pytest files should include:

```python
import pytest

NUM_GPUS = 0

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
```

- `run-ci-changed` extracts a top-level `NUM_GPUS = <N>` constant from added/modified `tests/test_*.py` and `tests/plugin_contracts/test_*.py`; if missing, it defaults to 8 GPUs. Set `NUM_GPUS = 0` for CPU-only tests.
- For GPU/e2e tests, follow the nearby file pattern (`prepare()`, `execute()`, `NUM_GPUS`, and any model/dataset constants).

## Local validation

- Run the exact existing test files you changed, if any.
- Run repository-wide checks only when they are already part of the task or workflow.
- Report the commands executed, results, and GPU requirements or untested paths.

## Workflow generation

For CI workflow changes:

1. Edit `.github/workflows/pr-test.yml.j2`
2. Regenerate workflows:

```bash
python .github/workflows/generate_github_workflows.py
```

3. Include both the template and generated workflow file in the change set (`.j2` and `.yml`). If the user asked for a commit, commit both.

## Reference Locations

- Pytest config: `pyproject.toml`
- Tests: `tests/`
- CI template: `.github/workflows/pr-test.yml.j2`
- CI guide: `docs/en/developer_guide/ci.md`
