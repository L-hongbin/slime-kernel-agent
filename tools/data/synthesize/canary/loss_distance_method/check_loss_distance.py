#!/usr/bin/env python3
"""Offline registry check for the loss/distance canary generator."""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.synthesize.canary.loss_distance_method import generate_loss_distance as generator  # noqa: E402


def main() -> None:
    references: set[str] = set()
    ast_hashes: set[str] = set()
    families: collections.Counter[str] = collections.Counter()
    for spec in generator.TEMPLATES:
        for variant in range(spec.variants):
            code, _, input_contract = generator._source(spec, variant)
            static = generator._static_contract(code, spec, input_contract)
            manifest = {
                "template_id": spec.template_id,
                "template_variant": variant,
                "mode_behavior": "stateless",
            }
            replay = generator.replay_manifest(manifest)
            assert replay["code"] == code
            assert replay["input_contract"] == input_contract
            assert replay["static_proof"] == static
            references.add(hashlib.sha256(code.encode()).hexdigest())
            ast_hashes.add(generator._normalized_ast_sha256(code))
            families[spec.primary_family] += 1

    assert len(references) == len(ast_hashes) == generator.EXACT_ROWS
    assert dict(families) == {
        "loss_cross_entropy": 4,
        "loss_smooth_l1": 4,
        "loss_triplet_margin": 4,
    }
    print(json.dumps({"families": dict(families), "rows": generator.EXACT_ROWS, "status": "passed"}, sort_keys=True))


if __name__ == "__main__":
    main()
