#!/usr/bin/env python3
"""Offline contract check for prompt_tvm_v4 construction and augmentation."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pyarrow as pa

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.synthesize.augment_prompt_tasks import AugmentationPolicy, augment_row  # noqa: E402
from tools.data.synthesize.build_prompt_tvm_v4 import (  # noqa: E402
    MODE_CONTRACT_MARKER,
    PARTITIONS,
    _extend_schema,
    inject_mode_contract,
)


REFERENCE = """import torch
import torch.nn as nn

batch_size = 4
channels = 3
height = 8

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, target):
        return torch.relu(x)

def get_inputs():
    x = torch.randn(batch_size, channels, height, height)
    target = torch.randn(batch_size, channels, height, height)
    return [x, target]

def get_init_inputs():
    return []
"""


def _model_ast(code: str) -> str:
    tree = ast.parse(code)
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model")
    return ast.dump(model, include_attributes=False)


def main() -> None:
    assert sum(spec.expected_rows for spec in PARTITIONS) == 64_321
    assert sum(spec.expected_rows for spec in PARTITIONS if spec.include_in_review_train) == 64_315
    assert len({spec.relative_path for spec in PARTITIONS}) == len(PARTITIONS)

    prompt = [{"role": "user", "content": "Opening.\n\nReference code."}]
    transformed = inject_mode_contract(prompt)
    assert MODE_CONTRACT_MARKER in transformed[0]["content"]
    assert transformed[0]["content"].startswith("Opening.\n\n")
    assert transformed[0]["content"].endswith("Reference code.")

    messages = pa.list_(pa.struct([pa.field("content", pa.string()), pa.field("role", pa.string())]))
    schema = pa.schema(
        [
            pa.field("data_source", pa.string()),
            pa.field("prompt", messages),
            pa.field("ability", pa.string()),
            pa.field(
                "reward_model", pa.struct([pa.field("ground_truth", pa.string()), pa.field("style", pa.string())])
            ),
            pa.field("extra_info", pa.struct([pa.field("uuid", pa.string())])),
        ]
    )
    assert _extend_schema(schema).field("extra_info").type.get_field_index("v4") >= 0

    user_prompt = [{"role": "user", "content": "Optimize this model.\n\n" + REFERENCE}]
    parent = {
        "data_source": "offline_check",
        "prompt": user_prompt,
        "ability": "kernel_optimization",
        "reward_model": {"ground_truth": REFERENCE, "style": "rule"},
        "extra_info": {
            "entry_point": "Model",
            "level": "check",
            "module_name": "Model",
            "ops": json.dumps(["torch.relu"]),
            "original_prompt": user_prompt,
            "repo_name": "offline_check",
            "type": "offline_check",
            "uuid": "parent_fixture",
        },
    }
    policy = AugmentationPolicy(
        shape_scales=(2,),
        value_families=("randn", "rand"),
        dtype_targets=(),
        layout_targets=(),
        include_joint_cells=True,
    )
    children, decisions = augment_row(parent, source_artifact_sha256="a" * 64, source_row_index=7, policy=policy)
    assert len(children) == sum(decision["keep"] is True for decision in decisions)
    assert len({child["extra_info"]["uuid"] for child in children}) == len(children)
    for child in children:
        metadata = child["extra_info"]["augmentation"]
        assert metadata["parent_uuid"] == "parent_fixture"
        assert metadata["source_row_index"] == 7
        assert _model_ast(child["reward_model"]["ground_truth"]) == _model_ast(REFERENCE)

    print(json.dumps({"augmentation_rows": len(children), "partitions": len(PARTITIONS), "status": "passed"}))


if __name__ == "__main__":
    main()
