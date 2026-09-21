from __future__ import annotations

from pathlib import Path
from typing import Any

from ..model import MMDSValidationError


def resolve_video_source(value: Any) -> str:
    """Extract a path or URL from an MMDS video field value."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("source", "path", "uri"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
        raise MMDSValidationError(
            "Video media dictionaries must contain a 'source', 'path', or 'uri' string key."
        )
    raise MMDSValidationError(
        "Video media fields must be a string path/URL or a dictionary, "
        f"got {type(value).__name__!r}."
    )


def resolve_local_media_path(
    path_value: str | Path,
    *,
    media_root: str | Path,
) -> Path:
    """Resolve a local media path and require it to remain under ``media_root``."""
    root = Path(media_root).expanduser().resolve()
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise MMDSValidationError(
            f"Local media path {str(path_value)!r} is outside media_root {str(root)!r}."
        ) from exc

    if not resolved.is_file():
        raise MMDSValidationError(
            f"Local media path {str(resolved)!r} does not exist or is not a file."
        )
    return resolved
