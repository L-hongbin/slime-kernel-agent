#!/usr/bin/env python3
"""Allow an audited SGLang retract-pause to flush stale model state.

SGLang 0.5.16 parks retracted requests in ``waiting_queue`` after releasing
their request-owned KV/GDN state. Its generic ``flush_cache`` guard still
requires that queue to be empty, so the weight-refit path cannot invalidate
RadixCache entries produced by the old weights. This fail-closed source patch
implements the narrow upstream fix proposed in sglang#33784: ignore only the
waiting-queue term while the scheduler is paused and every active-state idle
check still passes.

Unknown, unpatched upstream layouts are rejected instead of being rewritten.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SCHEDULER_PATH = Path("/usr/local/lib/python3.12/dist-packages/sglang/srt/managers/scheduler.py")
_MARKER = "# slime: retract-pause may flush after active KV/GDN owners drain."


@dataclass(frozen=True)
class Replacement:
    old: str
    new: str
    count: int = 1


_REPLACEMENTS = (
    Replacement(
        "    def is_fully_idle(self, for_health_check=False) -> bool:\n",
        "    def is_fully_idle(self, for_health_check=False, ignore_waiting_queue=False) -> bool:\n",
    ),
    Replacement(
        "        # Waiting queues: waiting + bootstrapping + preallocation + kv transfer (decode)\n"
        "        idle &= len(self.waiting_queue) == 0\n",
        "        # Waiting queues: waiting + bootstrapping + preallocation + kv transfer (decode)\n"
        f"        {_MARKER}\n"
        "        if not ignore_waiting_queue:\n"
        "            idle &= len(self.waiting_queue) == 0\n",
    ),
    Replacement(
        "        if self.is_fully_idle():\n" "            self.cur_batch_for_debug = None\n",
        f"        {_MARKER}\n"
        "        flushable = self.is_fully_idle() or (\n"
        "            self._engine_paused and self.is_fully_idle(ignore_waiting_queue=True)\n"
        "        )\n"
        "        if flushable:\n"
        "            self.cur_batch_for_debug = None\n",
    ),
)


def _state(source: str) -> str:
    old_counts = [source.count(item.old) for item in _REPLACEMENTS]
    new_counts = [source.count(item.new) for item in _REPLACEMENTS]
    expected = [item.count for item in _REPLACEMENTS]
    marker_count = source.count(_MARKER)
    if old_counts == expected and marker_count == 0:
        if "def pause_generation" not in source or "retract_all(" not in source:
            raise RuntimeError("SGLang scheduler lacks the audited retract-pause implementation")
        return "unpatched"
    if old_counts == [0] * len(_REPLACEMENTS) and new_counts == expected and marker_count == 2:
        return "patched"
    raise RuntimeError(
        "SGLang scheduler does not match the audited retract-flush layout: "
        f"old_counts={old_counts}, expected_old={expected}, "
        f"new_counts={new_counts}, marker_count={marker_count}"
    )


def patch_file(path: Path = DEFAULT_SCHEDULER_PATH, *, check_only: bool = False) -> str:
    source = path.read_text(encoding="utf-8")
    state = _state(source)
    if check_only:
        if state != "patched":
            raise RuntimeError(f"SGLang retract-flush patch is not installed in {path}")
        return state

    if state == "unpatched":
        for item in _REPLACEMENTS:
            source = source.replace(item.old, item.new)
        if _state(source) != "patched":
            raise RuntimeError(f"SGLang retract-flush patch failed post-write verification for {path}")
        path.write_text(source, encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler-path", type=Path, default=DEFAULT_SCHEDULER_PATH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(f"SGLang retract-flush state: {patch_file(args.scheduler_path, check_only=args.check)}")


if __name__ == "__main__":
    main()
