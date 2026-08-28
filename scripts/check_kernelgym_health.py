#!/usr/bin/env python3
"""Standalone KernelGym health preflight used by DrKernel training jobs."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import aiohttp


DEFAULT_KERNELGYM_URL = "http://127.0.0.1:20211"
DEFAULT_TIMEOUT = 5.0
DEFAULT_ATTEMPTS = 3
DEFAULT_INTERVAL = 2.0
DEFAULT_BACKOFF_FACTOR = 1.0
DEFAULT_MAX_INTERVAL = 60.0


class KernelGymRequestError(RuntimeError):
    """Raised when KernelGym returns a non-2xx response or invalid payload."""


def _format_scalar(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _is_scalar(value: Any) -> bool:
    return not isinstance(value, (dict, list))


def _flatten_payload(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        if not value:
            return [(prefix or "value", "(empty object)")]

        rows = []
        for key in sorted(value):
            field = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_payload(value[key], field))
        return rows

    if isinstance(value, list):
        if not value:
            return [(prefix or "value", "(empty list)")]

        rows = []
        for index, item in enumerate(value):
            field = f"{prefix}[{index}]" if prefix else f"[{index}]"
            rows.extend(_flatten_payload(item, field))
        return rows

    return [(prefix or "value", _format_scalar(value))]


def _print_rows(title: str, headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(header), *(len(row[col_idx]) for row in rows)) for col_idx, header in enumerate(headers)]
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print(title)
    print(border)
    print("| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |")
    print(border)
    for row in rows:
        print("| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) + " |")
    print(border)


def _print_key_value_table(title: str, payload: Any) -> None:
    rows = [[field, value] for field, value in _flatten_payload(payload)]
    _print_rows(title, ["Field", "Value"], rows)


def _record_columns(records: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "available",
        "memory_used",
        "utilization_gpu_percent",
    ]
    hidden = {"memory_total", "memory_used_percent", "name", "source"}
    keys = {key for record in records for key, value in record.items() if _is_scalar(value) and key not in hidden}
    return [key for key in preferred if key in keys] + sorted(keys - set(preferred))


def _display_column(column: str) -> str:
    if column == "utilization_gpu_percent":
        return "util"
    return column


def _print_record_table(title: str, records_by_name: dict[str, dict[str, Any]]) -> None:
    records = list(records_by_name.values())
    columns = _record_columns(records)
    if not columns:
        _print_key_value_table(title, records_by_name)
        return

    row_header = "GPU" if "gpu" in title.lower() else "Name"
    headers = [row_header, *[_display_column(column) for column in columns]]
    rows = []
    for name in sorted(records_by_name):
        record = records_by_name[name]
        rows.append([name, *[_format_scalar(record.get(column)) for column in columns]])
    _print_rows(title, headers, rows)


def _as_record_mapping(payload: Any) -> dict[str, dict[str, Any]] | None:
    if isinstance(payload, dict) and payload and all(isinstance(value, dict) for value in payload.values()):
        return payload

    if isinstance(payload, list) and payload and all(isinstance(value, dict) for value in payload):
        return {str(idx): value for idx, value in enumerate(payload)}

    return None


def print_table(title: str, payload: Any) -> None:
    record_mapping = _as_record_mapping(payload)
    if record_mapping is not None:
        _print_record_table(title, record_mapping)
        return

    if not isinstance(payload, dict):
        _print_key_value_table(title, payload)
        return

    scalar_rows = [[key, _format_scalar(value)] for key, value in sorted(payload.items()) if _is_scalar(value)]
    if scalar_rows:
        _print_rows(title, ["Field", "Value"], scalar_rows)

    for key, value in sorted(payload.items()):
        if _is_scalar(value):
            continue

        record_mapping = _as_record_mapping(value)
        if record_mapping is not None:
            _print_record_table(key, record_mapping)
        else:
            _print_key_value_table(key, value)


@dataclass(frozen=True)
class KernelGymHealthClient:
    """Small async client matching slime_plugins.drkernel.kernelgym_rm health logic."""

    base_url: str
    timeout_s: float = 5.0
    session: aiohttp.ClientSession | None = None
    _owns_session: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        base_url = self.base_url.rstrip("/")
        if not base_url:
            raise ValueError("KernelGym base_url must not be empty")
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "_owns_session", self.session is None)

    def _get_session(self) -> aiohttp.ClientSession:
        session = self.session
        if session is not None:
            return session

        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        connector = aiohttp.TCPConnector(limit=8, enable_cleanup_closed=True)
        session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        object.__setattr__(self, "session", session)
        return session

    async def close(self) -> None:
        session = self.session
        if session is not None and self._owns_session and not session.closed:
            await session.close()

    async def request_json(self, method: str, path: str) -> dict[str, Any]:
        session = self._get_session()
        url = f"{self.base_url}{path}"
        request_timeout = aiohttp.ClientTimeout(total=self.timeout_s)

        try:
            async with session.request(method, url, timeout=request_timeout) as response:
                text = await response.text()
                if response.status >= 400:
                    raise KernelGymRequestError(
                        f"KernelGym {method} {path} failed with status {response.status}: {text[:1000]}"
                    )
                try:
                    payload = await response.json()
                except Exception as exc:
                    raise KernelGymRequestError(
                        f"KernelGym {method} {path} returned non-JSON payload: {text[:1000]}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise KernelGymRequestError(
                        f"KernelGym {method} {path} returned {type(payload).__name__}, expected object"
                    )
                return payload
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise KernelGymRequestError(f"KernelGym {method} {path} request failed: {exc}") from exc

    async def check_health(self) -> dict[str, Any]:
        try:
            payload = await self.request_json("GET", "/health")
        except KernelGymRequestError as exc:
            raise KernelGymRequestError(f"KernelGym health check failed for {self.base_url}: {exc}") from exc

        status = payload.get("status")
        if status != "healthy":
            raise KernelGymRequestError(f"KernelGym at {self.base_url} is not healthy (status={status!r}): {payload}")
        return payload


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight KernelGym /health before launching DrKernel training.")
    parser.add_argument(
        "--url",
        default=os.environ.get("KERNELGYM_URL", DEFAULT_KERNELGYM_URL),
        help=f"KernelGym base URL. Defaults to KERNELGYM_URL or {DEFAULT_KERNELGYM_URL}.",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Per-request timeout in seconds.")
    parser.add_argument(
        "--attempts",
        type=int,
        default=DEFAULT_ATTEMPTS,
        help="Number of health attempts. Use 0 to retry indefinitely.",
    )
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="Sleep seconds between attempts.")
    parser.add_argument(
        "--backoff-factor",
        type=float,
        default=DEFAULT_BACKOFF_FACTOR,
        help="Multiply the retry interval by this factor after each failure.",
    )
    parser.add_argument(
        "--max-interval",
        type=float,
        default=DEFAULT_MAX_INTERVAL,
        help="Maximum sleep seconds between health attempts.",
    )
    parser.add_argument(
        "--workers-status",
        action="store_true",
        help="After /health passes, also print GET /workers/status if the endpoint exists.",
    )
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    if args.attempts < 0:
        raise ValueError("--attempts must be >= 0")
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")
    if args.interval < 0:
        raise ValueError("--interval must be >= 0")
    if args.backoff_factor < 1:
        raise ValueError("--backoff-factor must be >= 1")
    if args.max_interval < 0:
        raise ValueError("--max-interval must be >= 0")

    client = KernelGymHealthClient(args.url, timeout_s=args.timeout)
    try:
        last_error: Exception | None = None
        attempt = 0
        retry_interval = min(args.interval, args.max_interval)
        while args.attempts == 0 or attempt < args.attempts:
            attempt += 1
            attempt_limit = "unbounded" if args.attempts == 0 else str(args.attempts)
            try:
                payload = await client.check_health()
                print(f"OK: KernelGym healthy at {client.base_url}")
                print_table("health", payload)

                if args.workers_status:
                    workers = await client.request_json("GET", "/workers/status")
                    print_table("workers/status", workers)
                return 0
            except Exception as exc:
                last_error = exc
                print(f"FAIL attempt {attempt}/{attempt_limit}: {exc}", file=sys.stderr)
                if args.attempts == 0 or attempt < args.attempts:
                    delay = retry_interval
                    print(
                        f"RETRY KernelGym health in {delay:g}s " f"(next attempt {attempt + 1}/{attempt_limit})",
                        file=sys.stderr,
                    )
                    await asyncio.sleep(delay)
                    retry_interval = min(retry_interval * args.backoff_factor, args.max_interval)

        print(f"KernelGym health check failed after {args.attempts} attempt(s): {last_error}", file=sys.stderr)
        return 1
    finally:
        await client.close()


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
