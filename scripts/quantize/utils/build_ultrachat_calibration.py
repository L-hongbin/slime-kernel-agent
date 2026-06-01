"""Build generic chat calibration JSONL from HuggingFaceH4/ultrachat_200k.

This is the preferred calibration source for SmoothQuant/GPTQ when evaluating
DrKernel/KernelBench, because it avoids using prompts or responses from the
target eval distribution.  The output schema is one JSON object per line with
`text` rendered through the target model's chat template.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", required=True, help="HF checkpoint dir whose tokenizer/chat template renders messages"
    )
    parser.add_argument("--output", required=True, type=Path, help="JSONL output path")
    parser.add_argument("--max-prompts", type=int, default=256)
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--split", default="train_sft")
    parser.add_argument(
        "--proxy",
        default=os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY"),
        help="Optional HTTP(S) proxy, e.g. http://192.168.28.186:7897",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    dataset = load_dataset(args.dataset, split=args.split, streaming=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    char_lens: list[int] = []
    with args.output.open("w") as f:
        for row in dataset:
            messages = row.get("messages")
            if not messages:
                continue
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            if not isinstance(text, str) or not text.strip():
                continue
            f.write(
                json.dumps(
                    {
                        "text": text,
                        "source": args.dataset,
                        "split": args.split,
                        "prompt_id": row.get("prompt_id"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
            char_lens.append(len(text))
            if count >= args.max_prompts:
                break

    if count < args.max_prompts:
        raise RuntimeError(f"only wrote {count} prompts, expected {args.max_prompts}")

    char_lens.sort()
    print(f"wrote {count} calibration prompts to {args.output}")
    print(
        "prompt char-len: "
        f"median={char_lens[len(char_lens) // 2]} "
        f"p90={char_lens[int(0.9 * len(char_lens))]} "
        f"max={max(char_lens)}"
    )


if __name__ == "__main__":
    main()
