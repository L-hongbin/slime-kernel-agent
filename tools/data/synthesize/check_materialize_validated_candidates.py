#!/usr/bin/env python3
"""Offline end-to-end check for the validated-candidate materializer."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.synthesize import materialize_validated_candidates as materializer  # noqa: E402


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha256(value: dict) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(encoded.encode())


def _ast_sha256(code: str) -> str:
    normalized = ast.dump(ast.parse(code), annotate_fields=True, include_attributes=False)
    return _sha256_bytes(normalized.encode())


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values), encoding="utf-8")


def _kernelgym_hashes() -> dict[str, str]:
    file_hashes = {f"{name}_sha256": _sha256_bytes(name.encode()) for name in materializer.KERNELGYM_EVALUATOR_FILES}
    bundle = {
        materializer.KERNELGYM_EVALUATOR_FILES[name]: file_hashes[f"{name}_sha256"]
        for name in sorted(materializer.KERNELGYM_EVALUATOR_FILES)
    }
    return {**file_hashes, "evaluator_bundle_sha256": _canonical_sha256(bundle)}


def _audit(candidate: Path, manifest: dict, *, passed: bool) -> dict:
    validator = _sha256_file(_REPO_ROOT / "tools/data/synthesize/validate_train_mode_contract.py")
    launcher = _sha256_file(_REPO_ROOT / "tools/data/synthesize/launch_reference_validation_shards.sh")
    kernelgym = _kernelgym_hashes()
    source = _sha256_file(candidate)
    payload = {
        "contract_version": materializer.REQUIRED_CONTRACT_VERSION,
        "source_sha256": source,
        "validator_source_sha256": validator,
        "launcher_source_sha256": launcher,
        **{f"kernelgym_{name}": value for name, value in kernelgym.items()},
        "code_column": "reward_model.ground_truth",
        "entry_point_column": "extra_info.entry_point",
        "mode_class_column": "extra_info.v4.mode_class",
        "expected_mode_class": None,
        "device": "cuda:0",
        "max_device_memory_gib": 64.0,
        "trials": 5,
        "seed": 42,
        "training": True,
        "paired_rng_seed_reset_required": True,
        "persistent_model_instances_required": True,
    }
    record = {
        "contract_version": materializer.REQUIRED_CONTRACT_VERSION,
        "contract_fingerprint": _canonical_sha256(payload),
        "contract_payload": payload,
        "validator_source_sha256": validator,
        "launcher_source_sha256": launcher,
        "source_sha256": source,
        "row_index": manifest["row_index"],
        "row_key": f"{manifest['row_index']}:{manifest['reference_sha256']}",
        "uuid": manifest["uuid"],
        "reference_sha256": manifest["reference_sha256"],
        "trials": 5,
        "seed": 42,
        "training": True,
        "device": "cuda:0",
        "max_device_memory_gib": 64.0,
        "passed": passed,
        "status": "passed" if passed else "timeout",
        "failure_reasons": [] if passed else ["worker_timeout"],
        "kernelgym": {"root": "/offline-check", "git_commit": "0" * 40, **kernelgym},
    }
    if not passed:
        return record
    record.update(
        {
            "kernelgym_compiled": True,
            "kernelgym_correctness": True,
            "persistent_model_instances": True,
            "reference_training": True,
            "identical_training": True,
            "reference_forward_calls": 5,
            "identical_forward_calls": 5,
            "kernelgym_metadata": {
                "correctness_forward_seed_reset_enabled": True,
                "correctness_trials_run": 5,
                "correctness_trials": "(5 / 5)",
                "correctness_budget_min_pass_trials": 5,
                "train_mode_validator_persistent_instances": True,
                "train_mode_validator_training": True,
                "train_mode_validator_contract": materializer.REQUIRED_CONTRACT_VERSION,
            },
            "gpu": {
                "device": "cuda:0",
                "name": "offline GPU",
                "compute_capability": [9, 0],
                "torch_version": "offline",
                "torch_cuda_version": "offline",
            },
            "memory_guard": {
                "enabled": True,
                "configured": True,
                "synchronized": True,
                "over_limit": False,
                "limit_bytes": materializer.REQUIRED_MEMORY_LIMIT_BYTES,
                "peak_allocated_bytes": 1,
                "peak_reserved_bytes": 1,
                "allocator_fraction": 0.5,
                "within_limit": True,
            },
        }
    )
    return record


def main() -> None:
    codes = [
        "import torch\nclass Model:\n    def __call__(self, x): return torch.relu(x)\n",
        "import torch\nclass Model:\n    def __call__(self, x): return torch.sigmoid(x)\n",
    ]
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        rows = [
            {
                "data_source": "offline_check",
                "reward_model": {"ground_truth": code, "style": "rule"},
                "extra_info": {"entry_point": "Model", "uuid": f"candidate-{index}"},
            }
            for index, code in enumerate(codes)
        ]
        candidate = root / "candidates.parquet"
        pq.write_table(pa.Table.from_pylist(rows), candidate)
        manifests = [
            {
                "row_index": index,
                "uuid": f"candidate-{index}",
                "reference_sha256": _sha256_bytes(code.encode()),
                "normalized_ast_sha256": _ast_sha256(code),
            }
            for index, code in enumerate(codes)
        ]
        manifest = root / "manifest.jsonl"
        audit = root / "audit.jsonl"
        _write_jsonl(manifest, manifests)
        _write_jsonl(
            audit, [_audit(candidate, manifests[0], passed=True), _audit(candidate, manifests[1], passed=False)]
        )
        output = root / "output"
        config = materializer.MaterializeConfig(
            input_parquet=candidate,
            generator_manifest=manifest,
            audit_paths=(audit,),
            output_parquet=output / "accepted.parquet",
            accepted_manifest=output / "accepted.jsonl",
            rejected_manifest=output / "rejected.jsonl",
            summary_json=output / "summary.json",
        )
        summary = materializer.materialize(config)
        assert summary["evidence_complete"] is True
        assert summary["decisions"]["accepted"] == 1
        assert summary["decisions"]["rejected"] == 1
        accepted = pq.read_table(config.output_parquet).to_pylist()
        assert [row["extra_info"]["uuid"] for row in accepted] == ["candidate-0"]
        assert "runtime_status:timeout" in config.rejected_manifest.read_text()
        print(json.dumps({"accepted": 1, "rejected": 1, "status": "passed"}, sort_keys=True))


if __name__ == "__main__":
    main()
