from __future__ import annotations

from collections.abc import Iterable, Iterator
from math import isfinite
from numbers import Real
from typing import Any

from ...model import DatasetExpr, MMDSValidationError, Row, SplitSpec


def split_clip_intervals(
    start: float,
    end: float,
    *,
    chunk_sec: float,
) -> list[tuple[float, float]]:
    """Return contiguous ``(chunk_start, chunk_end)`` pairs covering ``[start, end)``."""
    if (
        isinstance(chunk_sec, bool)
        or not isinstance(chunk_sec, Real)
        or not isfinite(float(chunk_sec))
        or float(chunk_sec) <= 0
    ):
        raise MMDSValidationError("chunk_sec must be a positive finite number.")
    if (
        isinstance(start, bool)
        or not isinstance(start, Real)
        or not isfinite(float(start))
        or isinstance(end, bool)
        or not isinstance(end, Real)
        or not isfinite(float(end))
    ):
        raise MMDSValidationError("clip start/end must be finite numbers.")
    start_f = float(start)
    end_f = float(end)
    if end_f < start_f:
        raise MMDSValidationError(
            "clip end must be greater than or equal to clip start."
        )
    if end_f == start_f:
        return []

    chunk = float(chunk_sec)
    intervals: list[tuple[float, float]] = []
    cursor = start_f
    while cursor < end_f:
        chunk_end = min(cursor + chunk, end_f)
        intervals.append((cursor, chunk_end))
        cursor = chunk_end
    return intervals


def _require_finite(value: Any, *, label: str) -> float:
    """Require a finite real number; reject bools, strings, and missing values."""
    if value is None or isinstance(value, bool) or not isinstance(value, Real) or not isfinite(
        float(value)
    ):
        raise MMDSValidationError(f"Split: {label} must be a finite number.")
    return float(value)


def resolve_clip_bounds(
    row: Row,
    *,
    video_field: str,
    raw_video: Any,
    duration_field: str = "duration_sec",
) -> tuple[float, float]:
    """Resolve absolute clip ``(start, end)`` seconds for a row's video field."""
    start = 0.0
    end: float | None = None

    if isinstance(raw_video, dict):
        if "start" in raw_video:
            start = _require_finite(raw_video.get("start"), label=f"{video_field}.start")
        if "end" in raw_video:
            end = _require_finite(raw_video.get("end"), label=f"{video_field}.end")
    elif isinstance(raw_video, str):
        pass
    else:
        raise MMDSValidationError(
            f"Split: {video_field!r} must be a VideoView dict or a string path."
        )

    if end is None:
        if duration_field not in row:
            raise MMDSValidationError(
                f"Split: {video_field!r} must include numeric 'end' or the row must "
                f"include {duration_field!r}."
            )
        duration = _require_finite(row.get(duration_field), label=duration_field)
        if duration < 0:
            raise MMDSValidationError(
                f"Split: {duration_field!r} must be a non-negative finite number."
            )
        end = start + duration

    if end < start:
        raise MMDSValidationError(
            f"Split: resolved clip end ({end}) must be >= start ({start})."
        )
    return start, end


def build_chunk_video_view(
    raw_video: Any,
    *,
    chunk_start: float,
    chunk_end: float,
) -> dict[str, Any]:
    """Build a narrowed VideoView dict for one chunk."""
    if isinstance(raw_video, dict):
        narrowed = dict(raw_video)
        narrowed["type"] = "VideoView"
        narrowed["start"] = chunk_start
        narrowed["end"] = chunk_end
        return narrowed

    if isinstance(raw_video, str):
        return {
            "type": "VideoView",
            "source": raw_video,
            "start": chunk_start,
            "end": chunk_end,
        }

    raise MMDSValidationError(
        "Split: video field must be a VideoView dict or a string path."
    )


def split_row(row: Row, spec: SplitSpec) -> list[Row]:
    """Expand one input row into one row per fixed-duration video chunk.

    Overwrites ``spec.video_field`` with the narrowed chunk ``VideoView`` (the
    full-clip media is not preserved on chunk rows).
    """
    raw_video = row.get(spec.video_field)
    if raw_video is None:
        raise MMDSValidationError(
            f"Split: row is missing required video field {spec.video_field!r}."
        )

    doc_id = row.get(spec.doc_id_key)
    if not isinstance(doc_id, str) or not doc_id:
        raise MMDSValidationError(
            f"Split: row is missing non-empty string {spec.doc_id_key!r}."
        )

    clip_start, clip_end = resolve_clip_bounds(
        row,
        video_field=spec.video_field,
        raw_video=raw_video,
        duration_field=spec.duration_field,
    )
    intervals = split_clip_intervals(
        clip_start,
        clip_end,
        chunk_sec=spec.chunk_sec,
    )

    prefix = spec.output_prefix
    meta_keys = (
        f"{prefix}_id",
        f"{prefix}_chunk_num",
        f"{prefix}_chunk_start",
        f"{prefix}_chunk_end",
    )
    collisions = [key for key in meta_keys if key in row]
    if collisions:
        raise MMDSValidationError(
            "Split: refusing to overwrite existing row fields "
            f"{collisions!r}; choose a different output_prefix."
        )

    chunks: list[Row] = []
    for chunk_num, (chunk_start, chunk_end) in enumerate(intervals):
        chunk_row = dict(row)
        chunk_row[spec.video_field] = build_chunk_video_view(
            raw_video,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
        )
        chunk_row[meta_keys[0]] = doc_id
        chunk_row[meta_keys[1]] = chunk_num
        chunk_row[meta_keys[2]] = chunk_start
        chunk_row[meta_keys[3]] = chunk_end
        chunks.append(chunk_row)
    return chunks


def _apply_split(node: DatasetExpr, rows: Iterable[Row]) -> Iterator[Row]:
    spec = node.spec
    if not isinstance(spec, SplitSpec):
        raise MMDSValidationError("Split nodes require a SplitSpec.")

    for row in rows:
        yield from split_row(row, spec)
