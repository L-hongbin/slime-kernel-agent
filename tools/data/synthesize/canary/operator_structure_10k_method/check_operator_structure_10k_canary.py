#!/usr/bin/env python3
"""Offline registry and selection check for the operator/structure canary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.synthesize.canary.operator_structure_10k_method import (  # noqa: E402
    generate_operator_structure_10k as generator,
)
from tools.data.synthesize.canary.operator_structure_10k_method import (  # noqa: E402
    materialize_operator_structure_10k_canary as canary,
)


def _selection_record(index: int, manifest: dict) -> dict:
    calls = manifest["static_proof"]["complexity"]["forward_call_count"]
    bucket = "10-14" if calls <= 14 else "15-24" if calls <= 24 else "25+"
    return {
        "row_index": index,
        "uuid": manifest["uuid"],
        "reference_sha256": manifest["reference_sha256"],
        "template_id": manifest["template_id"],
        "template_variant": manifest["template_variant"],
        "primary_family": manifest["primary_family"],
        "coverage_labels": manifest["coverage_labels"],
        "mode_behavior": manifest["mode_behavior"],
        "top_level_class_count": manifest["static_proof"]["top_level_class_count"],
        "complexity_bucket": bucket,
        "forward_calls": calls,
        "manifest": manifest,
    }


def main() -> None:
    rows, manifests = generator.build_records(git_commit="0" * 40, generator_sha256="0" * 64)
    assert len(rows) == len(manifests) == generator.EXACT_CANDIDATE_ROWS
    by_template: dict[str, list[dict]] = {}
    for manifest in manifests:
        by_template.setdefault(manifest["template_id"], []).append(manifest)
    assert len(by_template) == 48
    for group in by_template.values():
        ordered = sorted(group, key=lambda item: item["template_variant"])
        for manifest in (ordered[0], ordered[len(ordered) // 2], ordered[-1]):
            replay = generator.replay_manifest(manifest)
            assert replay["code"] == rows[manifest["candidate_row_index"]]["reward_model"]["ground_truth"]

    selected, coverage = canary.select(
        [_selection_record(index, manifest) for index, manifest in enumerate(manifests)]
    )
    assert len(selected) == canary.ROWS
    assert set(coverage["per_template"].values()) == {canary.PER_TEMPLATE}
    assert coverage["required_v2_execution_cells"]["all_registry_templates"] == 48
    print(
        json.dumps(
            {"candidates": len(rows), "selected": len(selected), "templates": len(by_template), "status": "passed"},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
