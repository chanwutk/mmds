from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from ...model import DatasetExpr, MMDSValidationError, Row, SplitSpec


def split_clip_intervals(
    start: float,
    end: float,
    *,
    chunk_sec: float,
) -> list[tuple[float, float]]:
    """Return contiguous ``(chunk_start, chunk_end)`` pairs covering ``[start, end)``."""
    if chunk_sec <= 0:
        raise ValueError("chunk_sec must be positive.")
    if end < start:
        raise ValueError("clip end must be greater than or equal to clip start.")
    if end == start:
        return []

    intervals: list[tuple[float, float]] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + chunk_sec, end)
        intervals.append((cursor, chunk_end))
        cursor = chunk_end
    return intervals


def resolve_clip_bounds(
    row: Row,
    *,
    video_field: str,
    raw_video: Any,
) -> tuple[float, float]:
    """Resolve absolute clip ``(start, end)`` seconds for a row's video field."""
    start = 0.0
    end: float | None = None

    if isinstance(raw_video, dict):
        raw_start = raw_video.get("start")
        if isinstance(raw_start, (int, float)):
            start = float(raw_start)
        raw_end = raw_video.get("end")
        if isinstance(raw_end, (int, float)):
            end = float(raw_end)
    elif isinstance(raw_video, str):
        pass
    else:
        raise MMDSValidationError(
            f"Split: {video_field!r} must be a VideoView dict or a string path."
        )

    if end is None:
        duration_sec = row.get("duration_sec")
        if isinstance(duration_sec, (int, float)) and duration_sec >= 0:
            end = start + float(duration_sec)
        else:
            raise MMDSValidationError(
                f"Split: {video_field!r} must include numeric 'end' or the row must "
                "include 'duration_sec'."
            )

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
        narrowed["start"] = chunk_start
        narrowed["end"] = chunk_end
        if "type" not in narrowed:
            narrowed["type"] = "VideoView"
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
    """
    Expand one input row into one row per fixed-duration video chunk.
    Used by `Split` nodes to chunk video fields into fixed-duration chunks.
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
    )
    intervals = split_clip_intervals(
        clip_start,
        clip_end,
        chunk_sec=spec.chunk_sec,
    )

    prefix = spec.output_prefix
    chunks: list[Row] = []
    for chunk_num, (chunk_start, chunk_end) in enumerate(intervals):
        chunk_row = dict(row)
        chunk_row[spec.video_field] = build_chunk_video_view(
            raw_video,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
        )
        chunk_row[f"{prefix}_id"] = doc_id
        chunk_row[f"{prefix}_chunk_num"] = chunk_num
        chunk_row[f"{prefix}_chunk_start"] = chunk_start
        chunk_row[f"{prefix}_chunk_end"] = chunk_end
        chunks.append(chunk_row)
    return chunks


def _apply_split(node: DatasetExpr, rows: Iterable[Row]) -> Iterator[Row]:
    spec = node.spec
    if not isinstance(spec, SplitSpec):
        raise MMDSValidationError("Split nodes require a SplitSpec.")

    for row in rows:
        yield from split_row(row, spec)
