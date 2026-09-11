"""Canonical media-value validation and provenance helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from .model import MMDSExecutionError, MMDSValidationError


def resolve_media_source(value: Any) -> str:
    """Return the path or URI carried by a supported video value."""
    if isinstance(value, str) and value:
        return value
    if isinstance(value, Mapping):
        for key in ("path", "uri", "source"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate:
                return candidate
        raise MMDSValidationError(
            "Video values must contain a non-empty 'path', 'uri', or 'source'."
        )
    raise MMDSValidationError(
        "Video values must be a non-empty path/URI string or a mapping."
    )


def local_media_path(source: str, *, base_path: Path | None = None) -> Path:
    """Resolve a local path or file URI, rejecting remote media references."""
    parsed = urlparse(source)
    if parsed.scheme == "file":
        path = Path(url2pathname(parsed.path))
    elif parsed.scheme:
        raise MMDSExecutionError(
            "Materialized View currently requires a local path or file URI."
        )
    else:
        path = Path(source)
        if not path.is_absolute() and base_path is not None:
            path = base_path / path
    path = path.resolve()
    if not path.is_file():
        raise MMDSExecutionError(f"View source video does not exist: {path}")
    return path


def media_fingerprint(value: Any, source_path: Path) -> str:
    """Build a stable execution-time identity for a local media value."""
    if isinstance(value, Mapping):
        sha256 = value.get("sha256")
        if isinstance(sha256, str) and sha256:
            if len(sha256) != 64 or any(
                character not in "0123456789abcdefABCDEF" for character in sha256
            ):
                raise MMDSValidationError(
                    "Video sha256 metadata must contain 64 hexadecimal characters."
                )
            return f"sha256:{sha256}"
    stat = source_path.stat()
    payload = {
        "path": str(source_path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "stat:" + hashlib.sha256(encoded).hexdigest()


def materialized_video_value(
    value: Any,
    path: Path,
    *,
    start: float,
    end: float,
    sha256: str,
) -> dict[str, Any]:
    """Create a standalone Video value while retaining source provenance."""
    result: dict[str, Any] = {
        "type": "Video",
        "path": str(path),
        "mime_type": "video/mp4",
        "sha256": sha256,
        "source_start_seconds": start,
        "source_end_seconds": end,
    }
    if isinstance(value, Mapping):
        for field in ("source_id", "fps"):
            if field in value:
                result[field] = value[field]
        source_sha256 = value.get("sha256")
        if isinstance(source_sha256, str) and source_sha256:
            result["source_sha256"] = source_sha256
    return result


__all__ = [
    "local_media_path",
    "materialized_video_value",
    "media_fingerprint",
    "resolve_media_source",
]
