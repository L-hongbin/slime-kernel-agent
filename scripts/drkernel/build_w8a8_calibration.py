"""Extract rendered DrKernel prompts from a prior eval dump for W8A8 calibration.

Reads `<run>/dumps/rollout_data/eval_0.pt`, pulls every turn's
`prompt_snapshot` field, and emits a JSONL where each line is
`{"text": <prompt>, "turn": <int>, "problem_id": <id>}`. The output is
consumed by `quantize_w8a8.py --calibration-path` for GPTQ calibration.

Source dump should be representative of the production rollout
distribution (same prompt template + same model family + same KG
backend). v2_3 / current default template eval dumps are good
candidates.
"""

from __future__ import annotations

import argparse
import json

import torch


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--eval-pt",
        required=True,
        help="Source eval_0.pt under <run>/dumps/rollout_data/",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="JSONL output path (one prompt per line, key=text)",
    )
    ap.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Optional cap on number of prompts written (default: all turns)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    d = torch.load(args.eval_pt, map_location="cpu", weights_only=False)
    samples = d["samples"]
    print(f"loaded {len(samples)} samples from {args.eval_pt}")
    count = 0
    with open(args.output, "w") as f:
        for s in samples:
            turns = (s.get("metadata") or {}).get("turns") or []
            for ti, t in enumerate(turns):
                text = t.get("prompt_snapshot")
                if not text or not isinstance(text, str):
                    continue
                f.write(
                    json.dumps(
                        {
                            "text": text,
                            "turn": ti,
                            "problem_id": (s.get("metadata") or {}).get("problem_id"),
                        }
                    )
                    + "\n"
                )
                count += 1
                if args.max_prompts is not None and count >= args.max_prompts:
                    break
            if args.max_prompts is not None and count >= args.max_prompts:
                break
    print(f"wrote {count} calibration prompts to {args.output}")
    lens = []
    with open(args.output) as f:
        for line in f:
            lens.append(len(json.loads(line)["text"]))
    if lens:
        lens.sort()
        print(f"prompt char-len: median={lens[len(lens) // 2]} " f"p90={lens[int(0.9 * len(lens))]} max={max(lens)}")


if __name__ == "__main__":
    main()
