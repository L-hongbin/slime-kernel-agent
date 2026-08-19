"""Small, side-effect-free helpers for rendering configuration in logs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


REDACTED_VALUE = "<redacted>"


def _name_parts(name: object) -> tuple[str, ...]:
    """Normalize snake/kebab/camel-case names into uppercase components."""
    text = str(name)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    return tuple(part for part in re.split(r"[^A-Za-z0-9]+", text.upper()) if part)


def is_sensitive_config_key(name: object) -> bool:
    """Return whether a configuration key conventionally carries a secret."""
    parts = _name_parts(name)
    if any(part in {"APIKEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "CREDENTIALS"} for part in parts):
        return True

    pairs = set(zip(parts, parts[1:], strict=False))
    return bool(
        pairs
        & {
            ("API", "KEY"),
            ("ACCESS", "KEY"),
            ("PRIVATE", "KEY"),
            ("WANDB", "KEY"),
        }
    )


def redact_secrets_for_logging(value: Any, *, key: object | None = None) -> Any:
    """Return a recursively redacted logging view without mutating ``value``.

    Mapping keys and container layout are retained so non-secret configuration
    remains reviewable. Values whose key looks credential-bearing are replaced
    by a constant marker that does not reveal the original value or its length.
    """
    if key is not None and is_sensitive_config_key(key):
        return REDACTED_VALUE

    if isinstance(value, Mapping):
        return {
            nested_key: redact_secrets_for_logging(nested_value, key=nested_key)
            for nested_key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets_for_logging(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets_for_logging(item) for item in value)

    return value
