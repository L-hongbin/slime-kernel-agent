# Source after HF_MODEL_PATH is resolved. Only inspect safetensors headers;
# never load model tensors or infer weight presence from config.json.
SGLANG_SPECULATIVE_ARGS=()
MTP_WEIGHT_COUNT=$(python3 - "${HF_MODEL_PATH}" <<'PY'
import json
import sys
from pathlib import Path

from safetensors import safe_open

model_dir = Path(sys.argv[1])
index_path = model_dir / "model.safetensors.index.json"
if index_path.is_file():
    weight_map = json.loads(index_path.read_text())["weight_map"]
    files = sorted({model_dir / filename for filename in weight_map.values()})
else:
    files = sorted(model_dir.glob("*.safetensors"))

count = 0
for path in files:
    with safe_open(path, framework="np") as weights:
        count += sum("mtp" in name.split(".") or "mtp_layers" in name.split(".") for name in weights.keys())
print(count)
PY
) || return 1

if (( MTP_WEIGHT_COUNT > 0 )); then
    echo "Detected ${MTP_WEIGHT_COUNT} MTP weights in ${HF_MODEL_PATH}; enabling EAGLE speculative decoding."
    SGLANG_SPECULATIVE_ARGS=(
        --sglang-speculative-algorithm EAGLE
        --sglang-speculative-num-steps 3
        --sglang-speculative-eagle-topk 1
        --sglang-speculative-num-draft-tokens 4
    )
else
    echo "WARNING: No MTP weights found in ${HF_MODEL_PATH}; speculative decoding is disabled." >&2
    echo "When converting with tools/convert_torch_dist_to_hf.py, add --add-missing-from-origin-hf (-a) and --origin-hf-dir /path/to/original_hf_with_mtp." >&2
    echo "This fills missing weights from the original HF model, which must itself contain MTP weights; it does not recover trained MTP weights absent from the source checkpoint." >&2
    echo "The parallel converter currently does not support --add-missing-from-origin-hf; use the converter above when filling missing weights." >&2
fi
