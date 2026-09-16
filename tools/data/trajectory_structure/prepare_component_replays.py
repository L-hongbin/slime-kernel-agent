"""Export selected saved submissions for isolated KernelGym component tracing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .analyze import file_sha

# Purposeful validation cases, not a prevalence sample. All turns are 1-based.
CASES = [
    ("qwen38", 267, 3, "looped_launches"),
    ("qwen38", 267, 4, "cooperative_launch"),
    ("qwen38", 259, 4, "concat_sgemm"),
    ("qwen38", 259, 5, "concat_tf32"),
    ("qwen38", 23, 4, "init_math_policy"),
    ("qwen38", 118, 4, "partial_revert"),
    ("dsv4", 304, 2, "in_kernel_recurrence"),
    ("dsv4", 304, 5, "precomputed_projection"),
    ("dsv4", 180, 5, "persistent_workspace"),
    ("dsv4", 31, 5, "descriptor_cache"),
    ("dsv4", 96, 5, "retained_binding_fix"),
    ("dsv4", 268, 5, "launch_configuration_repair"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--structure-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    wanted = {(m, str(g)): (g, t, purpose) for m, g, t, purpose in CASES}
    structures = {}
    for path in sorted((args.structure_root / "trajectories").glob("*.json")):
        structure = json.loads(path.read_text())
        identity = structure["identity"]
        key = (identity[0], identity[-1])
        if key in wanted:
            structures[key] = (path, structure)
    manifest = []
    for model, group, turn, purpose in CASES:
        path, structure = structures[(model, str(group))]
        selected = next(t for t in structure["turns"] if t["turn_idx"] == turn - 1)
        if not selected["complete_sections"]:
            raise ValueError(f"incomplete selected source: {model} {group} {turn}")
        raw_path = args.raw_root / model / f"g{group:03}.json"
        raw = next(t for t in json.loads(raw_path.read_text()) if t["turn_idx"] == turn - 1)
        response_sha = hashlib.sha256(raw["response"].encode()).hexdigest()
        assert response_sha == selected["observation"]["response_sha256"]
        custom_code = "\n\n".join(
            f"### {section}\n```{lang}\n{selected['sections'][section]['source']}\n```"
            for section, lang in [("CUDA_KERNELS", "cpp"), ("APPLY_BINDINGS", "cpp"), ("MODEL_NEW", "python")]
        )
        payload = {
            "identity": {
                "model": model,
                "group": group,
                "turn": turn,
                "trajectory_id": path.stem,
                "purpose": purpose,
                "historical_speedup": raw.get("speedup"),
            },
            "custom_code": custom_code,
            "reference_code": raw["label"]["ground_truth"],
            "provenance": {
                "response_sha256": response_sha,
                "raw_sha256": file_sha(raw_path),
                "structure_sha256": file_sha(path),
            },
            "historical_correct": raw.get("correct"),
            "historical_compiled": raw.get("compiled"),
        }
        output = args.output / f"{model}_g{group:03}_t{turn}.json"
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        manifest.append(
            {
                "file": output.name,
                "sha256": file_sha(output),
                **payload["identity"],
                "historical_correct": raw.get("correct"),
            }
        )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {"cases": len(manifest), "historically_correct": sum(x["historical_correct"] is True for x in manifest)}
        )
    )


if __name__ == "__main__":
    main()
