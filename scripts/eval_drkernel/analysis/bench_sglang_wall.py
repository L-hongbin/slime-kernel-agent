#!/usr/bin/env python3
"""Small HTTP wall-time benchmark for a running SGLang server.

It sends fixed-shape chat-completions requests concurrently and reports
completion-token throughput. Use it for quick decode A/B checks; it does not
exercise slime, KernelGym, or multi-turn orchestration.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from statistics import fmean

import aiohttp


def make_prompt(idx: int, words: int) -> str:
    prefix = "Long deterministic prompt for SGLang wall-time measurement."
    body = " ".join(f"sample_{idx}_word_{i}" for i in range(words))
    return f"{prefix}\n{body}"


async def fire(
    session: aiohttp.ClientSession,
    base_url: str,
    idx: int,
    prompt_words: int,
    max_tokens: int,
) -> dict:
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": make_prompt(idx, prompt_words)}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
        "ignore_eos": True,
    }
    t0 = time.time()
    try:
        async with session.post(f"{base_url}/v1/chat/completions", json=payload, timeout=1800) as resp:
            data = await resp.json()
            usage = data.get("usage", {})
            return {
                "idx": idx,
                "status": resp.status,
                "duration_s": time.time() - t0,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "error": data.get("error"),
            }
    except Exception as exc:  # noqa: BLE001
        return {"idx": idx, "duration_s": time.time() - t0, "error": repr(exc)}


async def run(args: argparse.Namespace) -> None:
    results: list[dict] = []
    t0 = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(
                fire(
                    session=session,
                    base_url=args.base_url.rstrip("/"),
                    idx=i,
                    prompt_words=args.prompt_words,
                    max_tokens=args.output_tokens,
                )
            )
            for i in range(args.concurrency)
        ]
        for task in asyncio.as_completed(tasks):
            results.append(await task)
    wall = time.time() - t0

    completed = [r for r in results if r.get("completion_tokens")]
    errored = [r for r in results if r not in completed]
    completion_tokens = sum(int(r.get("completion_tokens", 0)) for r in completed)
    prompt_tokens = sum(int(r.get("prompt_tokens", 0)) for r in completed)
    durations = [float(r["duration_s"]) for r in completed]
    summary = {
        "label": args.label,
        "base_url": args.base_url,
        "concurrency": args.concurrency,
        "prompt_words": args.prompt_words,
        "output_tokens_target": args.output_tokens,
        "n_completed": len(completed),
        "n_errors": len(errored),
        "wall_s": round(wall, 3),
        "prompt_tokens_total": prompt_tokens,
        "completion_tokens_total": completion_tokens,
        "throughput_completion_tok_s": round(completion_tokens / wall, 3) if wall > 0 else 0.0,
        "mean_req_s": round(fmean(durations), 3) if durations else None,
        "max_req_s": round(max(durations), 3) if durations else None,
        "errors": errored[:5],
    }
    print(json.dumps(summary, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--label", default="bench")
    parser.add_argument("--concurrency", type=int, default=96)
    parser.add_argument("--prompt-words", type=int, default=900)
    parser.add_argument("--output-tokens", type=int, default=512)
    args = parser.parse_args()
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
