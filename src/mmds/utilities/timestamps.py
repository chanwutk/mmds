"""ISO-8601 timestamp parsing helpers shared by case studies and UDFs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..model import MMDSValidationError


def parse_iso_timestamp(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp string to a timezone-aware UTC datetime.

    Returns ``None`` for missing/blank/non-string values or unparseable text.
    Naive timestamps are treated as UTC. A trailing ``Z`` is accepted.
    """
    if not isinstance(value, str) or not value:
        return None
    timestamp = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def require_iso_timestamp(value: Any, *, label: str = "timestamp") -> datetime:
    """Like :func:`parse_iso_timestamp`, but raise on missing or invalid input."""
    parsed = parse_iso_timestamp(value)
    if parsed is None:
        raise MMDSValidationError(
            f"{label} must be a valid ISO-8601 timestamp string, got {value!r}."
        )
    return parsed


__all__ = [
    "parse_iso_timestamp",
    "require_iso_timestamp",
]
