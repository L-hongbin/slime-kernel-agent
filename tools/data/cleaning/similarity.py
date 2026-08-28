"""Token-level similarity checks for benchmark decontamination.

The representation intentionally follows the Python-token Jaccard method used
for PyTorch-reference decontamination in recent kernel-generation work.  It is
small, deterministic, and auditable; unlike an embedding model it does not
require a network-fetched checkpoint during a production dataset build.
"""

from __future__ import annotations

import io
import tokenize
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

PythonToken = tuple[int, str]


@dataclass(frozen=True)
class TokenJaccardBaseline:
    dataset_index: int
    row_index: int
    tokens: frozenset[PythonToken]


@dataclass(frozen=True)
class TokenJaccardMatch:
    dataset_index: int
    row_index: int
    similarity: float


def python_token_set(code: str) -> frozenset[PythonToken]:
    """Return semantic Python lexical tokens, excluding layout and comments."""

    kept_types = {tokenize.NAME, tokenize.NUMBER, tokenize.STRING, tokenize.OP}
    return frozenset(
        (token.type, token.string)
        for token in tokenize.generate_tokens(io.StringIO(code).readline)
        if token.type in kept_types
    )


def token_jaccard(left: frozenset[PythonToken], right: frozenset[PythonToken]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _nested(row: dict[str, Any], path: str) -> Any:
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def read_token_jaccard_baselines(
    paths: Iterable[Path],
    *,
    code_key: str,
) -> list[TokenJaccardBaseline]:
    baselines: list[TokenJaccardBaseline] = []
    root = code_key.split(".", 1)[0]
    for dataset_index, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        row_index = 0
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=4096,
            columns=[root],
            use_threads=False,
        ):
            for row in batch.to_pylist():
                code = _nested(row, code_key)
                if isinstance(code, str):
                    try:
                        tokens = python_token_set(code)
                    except (IndentationError, SyntaxError, tokenize.TokenError):
                        tokens = frozenset()
                    if tokens:
                        baselines.append(TokenJaccardBaseline(dataset_index, row_index, tokens))
                row_index += 1
    return baselines


def best_token_jaccard_match(
    code: str,
    baselines: Iterable[TokenJaccardBaseline],
    *,
    threshold: float,
) -> TokenJaccardMatch | None:
    """Return the strongest baseline match strictly above ``threshold``."""

    tokens = python_token_set(code)
    best: TokenJaccardMatch | None = None
    for baseline in baselines:
        # Jaccard cannot exceed the ratio of the smaller to larger set.  This
        # cheap bound avoids most intersections for differently sized programs.
        smaller = min(len(tokens), len(baseline.tokens))
        larger = max(len(tokens), len(baseline.tokens))
        if larger and smaller / larger <= threshold:
            continue
        similarity = token_jaccard(tokens, baseline.tokens)
        if similarity <= threshold:
            continue
        if best is None or similarity > best.similarity:
            best = TokenJaccardMatch(
                baseline.dataset_index,
                baseline.row_index,
                similarity,
            )
    return best
