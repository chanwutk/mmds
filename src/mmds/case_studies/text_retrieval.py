"""Reusable helpers for the UCA ground-truth caption-to-video example."""

from __future__ import annotations

from typing import Any


def tag_gallery_video(row: dict[str, Any]) -> dict[str, Any]:
    """Add ``source_video_id`` on gallery rows."""
    if not isinstance(row, dict):
        return {}
    video_id = row.get("video_id")
    if not isinstance(video_id, str) or not video_id:
        return {}
    return {"source_video_id": video_id}


def _as_nonneg_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _video_source_ref(raw_video: Any) -> dict[str, Any] | None:
    if not isinstance(raw_video, dict):
        return None
    ref = {
        key: value
        for key in ("path", "uri", "source")
        if isinstance((value := raw_video.get(key)), str) and value
    }
    return ref or None


def format_gt_caption_clip(row: dict[str, Any]) -> dict[str, Any]:
    """Materialize a ground-truth caption plus VideoView clip."""
    if not isinstance(row, dict):
        return {}
    left = row.get("left")
    right = row.get("right")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return {}
    start = _as_nonneg_float(left.get("start_sec"))
    end = _as_nonneg_float(left.get("end_sec"))
    if start is None or end is None or end < start:
        return {}
    media_ref = _video_source_ref(right.get("video"))
    if media_ref is None:
        return {}
    video_id = left.get("video_id")
    if not isinstance(video_id, str):
        video_id = right.get("video_id")
    return {
        "caption_id": left.get("caption_id"),
        "video_id": video_id,
        "caption": left.get("caption"),
        "start_sec": start,
        "end_sec": end,
        "timestamps": [start, end],
        "video": {"type": "VideoView", **media_ref, "start": start, "end": end},
        "title": right.get("title"),
        "duration_sec": right.get("duration_sec"),
    }


def group_gt_clips_by_video(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Group caption clips into an annotation_excerpt-style video row."""
    if not isinstance(rows, list):
        return {}
    clips: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        start = _as_nonneg_float(row.get("start_sec"))
        end = _as_nonneg_float(row.get("end_sec"))
        caption = row.get("caption")
        if start is None or end is None or end < start or not isinstance(caption, str):
            continue
        clips.append(row)
    if not clips:
        return {}
    clips.sort(
        key=lambda row: (
            float(row["start_sec"]),
            float(row["end_sec"]),
            str(row.get("caption_id") or ""),
        )
    )
    first = clips[0]
    duration = _as_nonneg_float(first.get("duration_sec"))
    aggregate: dict[str, Any] = {
        "duration": duration if duration is not None else 0.0,
        "timestamps": [
            [float(row["start_sec"]), float(row["end_sec"])] for row in clips
        ],
        "sentences": [row["caption"] for row in clips],
    }
    title = first.get("title")
    if isinstance(title, str) and title:
        aggregate["title"] = title
    media_ref = _video_source_ref(first.get("video"))
    if media_ref is not None:
        aggregate["video"] = {"type": "Video", **media_ref}
    return aggregate


__all__ = [
    "format_gt_caption_clip",
    "group_gt_clips_by_video",
    "tag_gallery_video",
]
